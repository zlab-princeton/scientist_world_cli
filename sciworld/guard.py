"""Host-side steps of a run that touch the workspace after the agent has.

The agent can write anywhere in its workspace, including symbolic links. The
launcher runs on the host with the user's permissions, so every step in which
it reads a bind-mount source from the workspace or writes into the workspace
goes through this module, which never follows a link out of the workspace::

    python -m sciworld.guard mounts WORKSPACE PATH...
        Print the PATHs (relative to the workspace) that exist, one per
        line. They are the ones the launcher mounts read-only. Exit status 3
        if one of them leads outside the workspace.

    python -m sciworld.guard prepare WORKSPACE
        Create the cache directories the container's environment points to.

    python -m sciworld.guard test-phase WORKSPACE SOURCE [--localize]
        Install the held-out evaluator from the environment source, and
        remove a scoring/scores_test.json that is older than it. Exit status
        4 if the source does not have the evaluator; the workspace is then
        left as it was.

    python -m sciworld.guard test-scores WORKSPACE
        Print "yes" if scoring/scores_test.json exists in the workspace, else
        "no". After `test-phase` that file can only come from the held-out
        evaluator, so it is what tells that the test phase did its job.

    python -m sciworld.guard done WORKSPACE [STATUS_FILE]
        Print "yes" if the status file holds {"done": true}, else "no".
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

from sciworld.fsutil import (
    copy_nofollow, copy_tree, ensure_real_dir, is_within, make_owner_writable,
    remove_tree,
)

EXIT_ESCAPE = 3
EXIT_NO_TEST_SLICE = 4


def existing_mounts(workspace: Path, relative_paths: list) -> list:
    """The *relative_paths* that exist in *workspace*, in order.

    Raises ``ValueError`` naming the first path that resolves to a location
    outside the workspace: mounting it would expose that location to the
    agent.
    """
    workspace = Path(workspace)
    found: list = []
    for rel in relative_paths:
        path = workspace / rel
        if not path.exists():
            # Absent, or a link that is broken on this machine (one written
            # for the container's paths, for instance): nothing to mount.
            continue
        if not is_within(path, workspace):
            raise ValueError(
                f"{rel} in the workspace is a symbolic link that leads outside "
                f"the workspace ({os.path.realpath(str(path))})"
            )
        found.append(rel)
    return found


def prepare_workspace(workspace: Path) -> None:
    """Create ``.cache/tmp``, where the container's ``TMPDIR`` points."""
    ensure_real_dir(Path(workspace), ".cache/tmp")


def has_test_slice(source: Path) -> bool:
    source = Path(source)
    return (source / "eval" / "test").is_dir() and (source / "scripts" / "evaluate_test.sh").is_file()


def has_test_scores(workspace: Path) -> bool:
    """True if the workspace holds ``scoring/scores_test.json``: a file, and
    one of the workspace, not something a link leads to elsewhere."""
    path = Path(workspace) / "scoring" / "scores_test.json"
    return path.is_file() and is_within(path, workspace)


def _remove_stale_test_scores(workspace: Path) -> None:
    """Remove a ``scoring/scores_test.json`` written before the held-out
    evaluator existed: whatever it holds, that evaluator did not produce it.

    A ``scoring`` that leads out of the workspace is left alone; nothing
    there is ours to remove, and :func:`has_test_scores` will not accept it.
    """
    scoring = Path(workspace) / "scoring"
    if not is_within(scoring, workspace):
        return
    stale = scoring / "scores_test.json"
    if stale.is_symlink() or stale.is_file():
        stale.unlink()
    elif stale.is_dir():
        remove_tree(stale)


def install_test_slice(workspace: Path, source: Path, localize: bool = False) -> None:
    """Replace ``eval/`` with the full evaluator of *source*, add
    ``scripts/evaluate_test.sh`` and remove test scores that predate them.

    ``status.json`` is the scientist's file and is not touched: an agent
    that set it to done and later finds it changed back cannot be expected to
    notice. The launcher tells that the test phase did its job from the
    scores the held-out evaluator writes (:func:`has_test_scores`).

    Raises ``FileNotFoundError`` when *source* has no held-out evaluator.
    Nothing in the workspace is changed in that case.
    """
    workspace = Path(workspace)
    source = Path(source)
    if not has_test_slice(source):
        raise FileNotFoundError(
            f"{source} has no held-out evaluator (eval/test and scripts/evaluate_test.sh)"
        )
    staging = workspace / f".eval.tmp-{os.getpid()}"
    remove_tree(staging)
    try:
        copy_tree(source / "eval", staging, root=source)
        make_owner_writable(staging)
        remove_tree(workspace / "eval")
        os.replace(str(staging), str(workspace / "eval"))
    finally:
        remove_tree(staging)

    scripts = ensure_real_dir(workspace, "scripts")
    script = source / "scripts" / "evaluate_test.sh"
    mode = stat.S_IMODE(script.stat().st_mode) | stat.S_IXUSR | stat.S_IRUSR | stat.S_IWUSR
    copy_nofollow(script, scripts / "evaluate_test.sh", mode)
    if not os.path.lexists(str(workspace / "scoring")):
        (workspace / "scoring").mkdir()
    if localize:
        from sciworld.localize import localize_workspace
        localize_workspace(workspace, ["eval", "scripts/evaluate_test.sh"])
    _remove_stale_test_scores(workspace)


def is_done(workspace: Path, status_file: str = "status.json") -> bool:
    try:
        data = json.loads((Path(workspace) / status_file).read_text())
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("done") is True


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if args else 2
    command, rest = args[0], args[1:]
    if command == "mounts" and rest:
        try:
            found = existing_mounts(Path(rest[0]), rest[1:])
        except ValueError as exc:
            print(f"sciworld.guard: {exc}", file=sys.stderr)
            return EXIT_ESCAPE
        for rel in found:
            print(rel)
        return 0
    if command == "prepare" and len(rest) == 1:
        prepare_workspace(Path(rest[0]))
        return 0
    if command == "test-phase" and len(rest) >= 2:
        try:
            install_test_slice(Path(rest[0]), Path(rest[1]), localize="--localize" in rest[2:])
        except FileNotFoundError as exc:
            print(f"sciworld.guard: {exc}", file=sys.stderr)
            return EXIT_NO_TEST_SLICE
        except (OSError, ValueError) as exc:
            print(f"sciworld.guard: could not install the held-out evaluator: {exc}", file=sys.stderr)
            return 1
        print("sciworld.guard: held-out evaluator installed")
        return 0
    if command == "test-scores" and len(rest) == 1:
        print("yes" if has_test_scores(Path(rest[0])) else "no")
        return 0
    if command == "done" and rest:
        print("yes" if is_done(Path(rest[0]), *(rest[1:2])) else "no")
        return 0
    print(f"sciworld.guard: unknown command or missing arguments: {' '.join(args)}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
