"""File-system helpers.

Three concerns:

* Environment repositories occasionally contain entries without owner write
  permission. Copied as they are, they could not be modified by the scientist,
  replaced by ``--overwrite``, or even deleted (``make_owner_writable``,
  ``remove_tree``).
* A repository is third-party content. Copying it must not pull files from
  outside the repository into a workspace through a symbolic link
  (``copy_tree``).
* A workspace is writable by the agent. Code that runs on the host and writes
  into a workspace afterwards must not be redirected by a symbolic link the
  agent planted (``is_within``, ``ensure_real_dir``, ``write_nofollow``,
  ``copy_nofollow``).
"""

from __future__ import annotations

import os
import shutil
import stat
import time
from pathlib import Path
from typing import Optional


# ── Permissions ───────────────────────────────────────────────────────────


def _unlock(path: str) -> None:
    try:
        mode = os.lstat(path).st_mode
        wanted = stat.S_IRWXU if stat.S_ISDIR(mode) else (stat.S_IRUSR | stat.S_IWUSR)
        if not stat.S_ISLNK(mode) and mode & wanted != wanted:
            os.chmod(path, stat.S_IMODE(mode) | wanted)
    except OSError:
        pass


def make_owner_writable(root: Path) -> None:
    """Give the owner read and write access to everything under *root*
    (and search access to directories). Other permission bits are kept."""
    root = str(root)
    _unlock(root)
    for current, dirs, files in os.walk(root):
        for name in dirs:
            _unlock(os.path.join(current, name))  # before os.walk descends into it
        for name in files:
            _unlock(os.path.join(current, name))


def remove_tree(path: Path) -> None:
    """Delete *path* whatever its permissions. A missing path is not an error.

    Symbolic links are removed, never followed.
    """
    path = Path(path)
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if not path.exists():
        return
    target = str(path)
    _unlock(target)
    for current, dirs, _files in os.walk(target):
        for name in dirs:
            _unlock(os.path.join(current, name))
    shutil.rmtree(target)


def sweep_stale_temp(parent: Path, max_age: float = 86400.0) -> None:
    """Remove ``.<name>.tmp-<pid>`` and ``.<name>.old-<pid>`` entries that an
    interrupted command left in *parent* more than *max_age* seconds ago."""
    try:
        entries = list(Path(parent).iterdir())
    except OSError:
        return
    now = time.time()
    for entry in entries:
        name = entry.name
        if not name.startswith(".") or not (".tmp-" in name or ".old-" in name):
            continue
        if not name.rsplit("-", 1)[-1].isdigit():
            continue
        try:
            if now - entry.lstat().st_mtime > max_age:
                remove_tree(entry)
        except OSError:
            pass


# ── Symbolic links ────────────────────────────────────────────────────────


def is_within(path, root) -> bool:
    """True if *path*, with every symbolic link resolved, is *root* or lies
    inside it."""
    real_root = os.path.realpath(str(root))
    real = os.path.realpath(str(path))
    return real == real_root or real.startswith(real_root.rstrip(os.sep) + os.sep)


def copy_tree(src: Path, dst: Path, root: Optional[Path] = None) -> list:
    """Copy the directory *src* to *dst* and return what was left out.

    Symbolic links are replaced by what they point to, provided the target
    exists and lies inside *root* (default: *src*). A link that is broken,
    points outside *root*, or forms a loop is left out, and so is anything
    that cannot be read, anything that is not a regular file or directory,
    and the destination itself should a link lead back to it.
    The returned list holds ``(path relative to src, reason)`` pairs.
    """
    src = Path(src)
    dst = Path(dst)
    root_real = os.path.realpath(str(root if root is not None else src))
    dst_real = os.path.realpath(str(dst))
    skipped: list = []

    def _inside(path: str) -> bool:
        real = os.path.realpath(path)
        return real == root_real or real.startswith(root_real.rstrip(os.sep) + os.sep)

    def _skip(path: str, reason: str) -> None:
        skipped.append((os.path.relpath(path, str(src)), reason))

    def _copy(source: str, target: str, ancestors: frozenset) -> None:
        try:
            info = os.lstat(source)
        except OSError as exc:
            _skip(source, f"unreadable ({exc.strerror or exc})")
            return
        if stat.S_ISLNK(info.st_mode):
            if not os.path.exists(source):
                _skip(source, "broken symbolic link")
                return
            if not _inside(source):
                _skip(source, "symbolic link to a location outside the environment")
                return
        if os.path.isdir(source):
            real = os.path.realpath(source)
            if real in ancestors:
                _skip(source, "symbolic link loop")
                return
            if real == dst_real or real.startswith(dst_real.rstrip(os.sep) + os.sep):
                _skip(source, "the copy's own destination")
                return
            try:
                names = sorted(os.listdir(source))
            except OSError as exc:
                _skip(source, f"unreadable ({exc.strerror or exc})")
                return
            os.makedirs(target, exist_ok=True)
            for name in names:
                _copy(os.path.join(source, name), os.path.join(target, name), ancestors | {real})
            try:
                shutil.copystat(source, target)
            except OSError:
                pass
        elif os.path.isfile(source):
            try:
                shutil.copy2(source, target)
            except OSError as exc:
                _skip(source, f"unreadable ({exc.strerror or exc})")
        else:
            _skip(source, "not a regular file")

    _copy(str(src), str(dst), frozenset())
    return skipped


def ensure_real_dir(root: Path, relative: str) -> Path:
    """Return ``root/relative`` as a real directory inside *root*.

    Every component is checked: one that is a symbolic link or a file is
    removed and replaced by a directory, so the result can never be a path
    that leads out of *root*.
    """
    current = Path(root)
    current.mkdir(parents=True, exist_ok=True)
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            current.unlink()
        if not current.exists():
            current.mkdir()
    return current


def write_nofollow(path: Path, data: bytes, mode: int = 0o644) -> None:
    """Write *data* to *path*, replacing whatever is there.

    A symbolic link at *path* is removed, never written through. The parent
    directory must already be trusted (see :func:`ensure_real_dir`).
    """
    path = Path(path)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        remove_tree(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(path), flags, mode)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        # fchmod acts on the file we opened, whatever the path names now.
        os.fchmod(handle.fileno(), mode)


def copy_nofollow(src: Path, dst: Path, mode: Optional[int] = None) -> None:
    """Copy the file *src* to *dst* without writing through a symbolic link.

    *mode* defaults to the permissions of *src*.
    """
    src = Path(src)
    if mode is None:
        mode = stat.S_IMODE(src.stat().st_mode)
    write_nofollow(dst, src.read_bytes(), mode)
