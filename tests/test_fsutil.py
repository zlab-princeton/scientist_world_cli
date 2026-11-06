import os
import stat

from sciworld.fsutil import make_owner_writable, remove_tree
from sciworld.package import package_environment


def locked_tree(root):
    """A tree with a read-only file inside read-only directories."""
    inner = root / "outer" / "inner"
    inner.mkdir(parents=True)
    (inner / "file.txt").write_text("x")
    (inner / "file.txt").chmod(0o444)
    inner.chmod(0o555)
    (root / "outer").chmod(0o500)
    return root


def test_remove_tree_deletes_read_only_directories(tmp_path):
    root = locked_tree(tmp_path / "tree")
    remove_tree(root)
    assert not root.exists()
    remove_tree(root)  # missing: not an error


def test_remove_tree_never_follows_symlinks(tmp_path):
    keep = tmp_path / "keep"
    keep.mkdir()
    (keep / "precious.txt").write_text("data")
    root = tmp_path / "tree"
    root.mkdir()
    (root / "link").symlink_to(keep)
    remove_tree(root)
    assert (keep / "precious.txt").read_text() == "data"
    link = tmp_path / "direct"
    link.symlink_to(keep)
    remove_tree(link)
    assert not link.is_symlink() and keep.is_dir()


def test_make_owner_writable_keeps_other_bits(tmp_path):
    root = locked_tree(tmp_path / "tree")
    script = root / "run.sh"
    os.chmod(root / "outer", 0o700)
    script.write_text("#!/bin/sh\n")
    script.chmod(0o555)
    os.chmod(root / "outer", 0o500)
    make_owner_writable(root)
    assert stat.S_IMODE(script.stat().st_mode) == 0o755
    assert stat.S_IMODE((root / "outer" / "inner" / "file.txt").stat().st_mode) == 0o644
    assert stat.S_IMODE((root / "outer").stat().st_mode) == 0o700


def test_a_repository_with_read_only_entries_gives_an_editable_workspace(env_source, tmp_path):
    rootfs = env_source / "environment" / "base_images" / "rootfs" / "var"
    rootfs.mkdir(parents=True)
    (rootfs / "state").write_text("x")
    rootfs.chmod(0o500)
    (env_source / "method" / "predict.py").chmod(0o444)
    try:
        ws = package_environment(env_source, tmp_path / "ws", "improve")
        (ws / "method" / "predict.py").write_text("changed\n")       # the scientist can edit
        (ws / "environment" / "base_images" / "rootfs" / "var" / "new").write_text("y")
        remove_tree(ws)
        assert not ws.exists()
    finally:
        rootfs.chmod(0o700)


def test_overwrite_replaces_a_workspace_with_read_only_entries(configured):
    import sciworld
    from conftest import ARXIV_ID
    ws = sciworld.research(id=ARXIV_ID, agent="codex", mode="improve", setup_only=True)["env_dir"]
    locked = ws / "method" / "cache"
    locked.mkdir()
    (locked / "weights.bin").write_text("w")
    locked.chmod(0o500)
    try:
        sciworld.research(id=ARXIV_ID, agent="codex", mode="improve", setup_only=True, overwrite=True)
    finally:
        if locked.exists():
            locked.chmod(0o700)
    assert not locked.exists() and (ws / "method" / "predict.py").exists()


# ── Symbolic links ──

def test_is_within(tmp_path):
    from sciworld.fsutil import is_within
    root = tmp_path / "root"
    (root / "a").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "in").symlink_to("a")
    (root / "out").symlink_to(outside)
    (root / "broken").symlink_to(tmp_path / "gone")
    assert is_within(root, root) and is_within(root / "a", root) and is_within(root / "in", root)
    assert is_within(root / "not-there-yet", root)
    assert not is_within(root / "out", root) and not is_within(root / "out" / "x", root)
    assert not is_within(root / "broken", root)
    assert not is_within(tmp_path / "rootsibling", root)   # a name that merely starts the same
    assert not is_within(root / ".." / "outside", root)


def test_copy_tree_reports_what_it_leaves_out(tmp_path):
    from sciworld.fsutil import copy_tree
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "f.txt").write_text("f")
    (src / "rel").symlink_to("sub/f.txt")
    (src / "up").symlink_to("..")             # leaves src, but stays inside root
    (src / "gone").symlink_to("nowhere")
    # `up` leads to the directory that holds both the source and the
    # destination: neither may be copied into the copy.
    skipped = copy_tree(src, tmp_path / "dst", root=tmp_path)
    assert (tmp_path / "dst" / "rel").read_text() == "f" and not (tmp_path / "dst" / "rel").is_symlink()
    assert ("gone", "broken symbolic link") in skipped
    assert ("up/src", "symbolic link loop") in skipped
    assert ("up/dst", "the copy's own destination") in skipped
    assert not (tmp_path / "dst" / "up" / "dst").exists()
    # With the default root (src itself), `up` leads outside.
    skipped = copy_tree(src, tmp_path / "dst2")
    assert ("up", "symbolic link to a location outside the environment") in skipped
    assert sorted(p.name for p in (tmp_path / "dst2").iterdir()) == ["rel", "sub"]


def test_ensure_real_dir_replaces_links_and_files_on_the_way(tmp_path):
    from sciworld.fsutil import ensure_real_dir
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.mkdir()
    (root / "a").symlink_to(outside)
    result = ensure_real_dir(root, "a/b")
    assert result == root / "a" / "b" and result.is_dir()
    assert not (root / "a").is_symlink() and list(outside.iterdir()) == []
    (root / "c").write_text("a file where a directory is needed")
    assert ensure_real_dir(root, "c").is_dir()
    assert ensure_real_dir(tmp_path / "new-root", "x/y").is_dir()   # the root is created too


def test_write_and_copy_nofollow(tmp_path):
    from sciworld.fsutil import copy_nofollow, write_nofollow
    victim = tmp_path / "victim"
    victim.write_text("mine")
    target = tmp_path / "target"
    target.symlink_to(victim)
    write_nofollow(target, b"new", 0o600)
    assert victim.read_text() == "mine" and target.read_bytes() == b"new" and not target.is_symlink()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    (tmp_path / "dirlink").symlink_to(tmp_path)
    write_nofollow(tmp_path / "dirlink", b"x")              # a link to a directory is replaced as well
    assert (tmp_path / "dirlink").read_bytes() == b"x" and victim.exists()
    src = tmp_path / "script.sh"
    src.write_text("#!/bin/bash\n")
    src.chmod(0o750)
    copy_nofollow(src, target)
    assert target.read_text() == "#!/bin/bash\n" and stat.S_IMODE(target.stat().st_mode) == 0o750


def test_sweep_stale_temp_removes_only_old_leftovers(tmp_path):
    import time
    from sciworld.fsutil import sweep_stale_temp
    old = time.time() - 3 * 86400
    parent = tmp_path / "paper"
    parent.mkdir()
    for name in (".source.tmp-123", ".run.old-456", ".run.tmp-notapid", "run", ".hidden"):
        (parent / name).mkdir()
        os.utime(parent / name, (old, old))
    (parent / ".fresh.tmp-789").mkdir()
    sweep_stale_temp(parent)
    assert sorted(p.name for p in parent.iterdir()) == [".fresh.tmp-789", ".hidden", ".run.tmp-notapid", "run"]
    sweep_stale_temp(tmp_path / "missing")  # not an error
