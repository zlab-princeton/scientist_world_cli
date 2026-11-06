"""Make a workspace runnable outside a container.

Environment scripts are written for the container, where the workspace is
mounted at ``/home/user``: they ``cd /home/user`` and pass paths such as
``/home/user/scoring/scores_train.json``. In venv mode there is no container,
so the workspace lives at its real path on this machine. This module rewrites
the container path to the real one in the workspace's text files.

Two kinds of file are told apart:

* Code, configuration and documentation: every ``/home/user`` is a path to
  use, so every occurrence is rewritten.
* Everything else under ``data/`` and ``eval/`` (tables, manifests, gold
  answers): a ``/home/user/<name>/...`` is rewritten only when ``<name>``
  exists in the workspace, which is what a reference to a workspace file looks
  like. Other occurrences are content and stay as they are.

It runs once when a venv-mode workspace is built, and again from the launcher
when the held-out evaluator is added::

    python -m sciworld.localize WORKSPACE [RELATIVE_PATH ...]
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

CONTAINER_HOME = "/home/user"

# ``/home/user`` as a path of its own: not the tail of a longer path or URL
# (``example.org/home/user``) and not the start of another name
# (``/home/username``, ``/home/user-data``, ``/home/user.name``). The group
# captures the first path component after it, if there is one.
_PATTERN = re.compile(
    rb"(?<![A-Za-z0-9_.~/])/home/user(?![A-Za-z0-9_-])(?!\.[A-Za-z0-9_])"
    rb"(?:/([A-Za-z0-9_.+@%=-]+))?"
)

# A real path is substituted into shell scripts, so it must not need quoting.
_SAFE_PATH = re.compile(r"^[A-Za-z0-9_./+@%-]+$")

# Never rewritten: the scientist's own tool state (at the top of the
# workspace), dependency and cache directories at any depth, and payloads
# fetched by scripts/download.sh.
_SKIP_TOP = {".git", ".venv", ".cache", ".local", ".config", ".npm", ".claude",
             ".codex", ".gemini", ".kimi", "trajectory"}
_SKIP_ANYWHERE = {".git", "__pycache__", "node_modules"}
_SKIP_PATHS = {("data", "downloads")}
_MAX_BYTES = 64 * 1024 * 1024

# Files in which a path is always something to run or to read as an instruction.
_CODE_SUFFIXES = {
    ".py", ".sh", ".bash", ".zsh", ".ksh", ".csh", ".fish",
    ".yaml", ".yml", ".toml", ".cfg", ".ini", ".conf", ".env", ".def", ".mk",
    ".r", ".jl", ".lua", ".pl", ".rb", ".js", ".mjs", ".ts", ".c", ".cc",
    ".cpp", ".h", ".hpp", ".java", ".go", ".rs", ".m", ".slurm", ".sbatch",
}
_CODE_NAMES = {"makefile", "dockerfile", "snakefile", "justfile"}
# Directories whose other files (tables, manifests, gold answers) are content.
_CONTENT_DIRS = {"data", "eval"}


def is_safe_workspace_path(workspace: Path) -> bool:
    """True if *workspace* can be substituted for ``/home/user`` in scripts."""
    return bool(_SAFE_PATH.match(str(workspace)))


def _eligible(relative: Path) -> bool:
    parts = relative.parts
    if not parts:
        return False
    if parts[0] in _SKIP_TOP or any(part in _SKIP_ANYWHERE for part in parts):
        return False
    return not any(parts[:len(skip)] == skip for skip in _SKIP_PATHS)


def _rewrites_everything(relative: Path, head: bytes) -> bool:
    """True if every occurrence in this file is a path to use."""
    if relative.suffix.lower() in _CODE_SUFFIXES or relative.name.lower() in _CODE_NAMES:
        return True
    if head.startswith(b"#!"):
        return True
    return not (relative.parts and relative.parts[0] in _CONTENT_DIRS)


def _rewrite_file(path: Path, relative: Path, workspace: Path, replacement: bytes) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        if path.stat().st_size > _MAX_BYTES:
            return False
        data = path.read_bytes()
    except OSError:
        return False
    if b"/home/user" not in data or b"\0" in data:
        return False

    everything = _rewrites_everything(relative, data[:2])
    known: dict = {}

    def _substitute(match) -> bytes:
        first = match.group(1)
        if not everything:
            if first is None:
                return match.group(0)
            name = first.decode("utf-8", errors="replace")
            if name not in known:
                known[name] = os.path.lexists(str(workspace / name))
            if not known[name]:
                return match.group(0)
        return replacement + match.group(0)[len(b"/home/user"):]

    new = _PATTERN.sub(_substitute, data)
    if new == data:
        return False
    try:
        mode = path.stat().st_mode
        path.write_bytes(new)
        path.chmod(mode)
    except OSError:
        return False
    return True


def localize_workspace(workspace: Path, subpaths=None, target: Optional[Path] = None) -> list:
    """Replace ``/home/user`` in the text files of *workspace*.

    The replacement is *target*, the path the workspace has (or will have)
    on this machine; by default the resolved path of *workspace* itself.
    *subpaths* limits the rewrite to those files or directories, relative to
    the workspace; by default the whole workspace is processed. Binary files,
    files over 64 MB and tool-state directories are left alone. Returns the
    rewritten paths, relative to the workspace.
    """
    workspace = Path(workspace).resolve()
    final = Path(target) if target is not None else workspace
    if not is_safe_workspace_path(final):
        raise ValueError(
            f"workspace path {str(final)!r} contains characters that are "
            "not safe to substitute into shell scripts; use a path made of "
            "letters, digits and ._/+@%-"
        )
    replacement = str(final).encode()
    roots = [workspace / p for p in subpaths] if subpaths else [workspace]
    changed: list = []
    for root in roots:
        if root.is_symlink():
            continue
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = []
            for current, dirs, files in os.walk(str(root)):
                here = Path(current)
                dirs[:] = sorted(d for d in dirs if _eligible((here / d).relative_to(workspace)))
                candidates += [here / name for name in sorted(files)]
        else:
            continue
        for path in candidates:
            relative = path.relative_to(workspace)
            if not _eligible(relative):
                continue
            if _rewrite_file(path, relative, workspace, replacement):
                changed.append(str(relative))
    return sorted(changed)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if args else 2
    try:
        changed = localize_workspace(Path(args[0]), args[1:] or None)
    except ValueError as exc:
        print(f"sciworld.localize: {exc}", file=sys.stderr)
        return 1
    print(f"sciworld.localize: rewrote {len(changed)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
