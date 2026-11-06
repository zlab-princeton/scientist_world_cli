import pytest

from sciworld import localize


def write(ws, files):
    for rel, data in files.items():
        path = ws / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else data.encode())


def test_rewrites_container_home_in_scripts(tmp_path):
    ws = tmp_path / "ws"
    (ws / "scripts").mkdir(parents=True)
    script = ws / "scripts" / "evaluate_train.sh"
    script.write_text(
        "cd /home/user\npython x.py --out /home/user/scoring/s.json\n"
        "ROOT=${ROOT:-/home/user/outputs}\nexport PATH=/home/user/.local/bin:$PATH\n"
        "echo /home/username /home/user-data /home/user.name /data/home/user\n"
        "curl https://example.org/home/user/guide\n"
        "# the workspace is at /home/user.\n"
    )
    script.chmod(0o755)
    assert localize.localize_workspace(ws) == ["scripts/evaluate_train.sh"]
    text = script.read_text()
    assert f"cd {ws}\n" in text and f"--out {ws}/scoring/s.json" in text
    # Paths that do not exist yet are rewritten too: this is code.
    assert f"ROOT=${{ROOT:-{ws}/outputs}}" in text and f"PATH={ws}/.local/bin:$PATH" in text
    # Other names, longer paths and URLs are not the container home.
    assert "/home/username /home/user-data /home/user.name /data/home/user" in text
    assert "https://example.org/home/user/guide" in text
    assert f"is at {ws}.\n" in text  # a full stop ends a sentence, not a name
    assert script.stat().st_mode & 0o777 == 0o755
    assert localize.localize_workspace(ws) == []  # idempotent


def test_data_files_keep_their_content_but_references_to_workspace_files_are_fixed(tmp_path):
    """A table of image paths must point at the images; a gold answer that
    happens to contain a path is content and must not change."""
    ws = tmp_path / "ws"
    write(ws, {
        "data/geo/img_1.png": b"\x89PNG\0",
        "data/geo/index.csv": "id,image_path\n1,/home/user/data/geo/img_1.png\n",
        "data/manifest.json": '{"run": "/home/user/data/rewrites/a.trec", "model": "/home/user/checkpoints/m"}',
        "data/prompts.json": '{"prompt": "cd /home/user and list /home/user/docs/report.txt"}',
        "eval/train/gold.json": '{"answer": "/home/user/docs/report.txt"}',
        "eval/train/evaluate.py": "GOLD = '/home/user/eval/train/gold.json'\nOUT = '/home/user/results/x'\n",
        "data/loader.py": "P = '/home/user/data'",
        "data/run": "#!/bin/bash\ncd /home/user/work\n",
        "data/notes": "see /home/user/work\n",
        "method/config.json": '{"out": "/home/user/outputs"}',
        "READMEs/problem.md": "Results go to /home/user/scoring.\n",
    })
    changed = localize.localize_workspace(ws)
    read = lambda rel: (ws / rel).read_text()
    assert read("data/geo/index.csv") == f"id,image_path\n1,{ws}/data/geo/img_1.png\n"
    # data/ exists in the workspace, checkpoints/ does not.
    assert read("data/manifest.json") == (
        f'{{"run": "{ws}/data/rewrites/a.trec", "model": "/home/user/checkpoints/m"}}')
    assert "data/prompts.json" not in changed and "eval/train/gold.json" not in changed
    assert read("eval/train/gold.json") == '{"answer": "/home/user/docs/report.txt"}'
    # Code and configuration: everything.
    assert read("eval/train/evaluate.py") == f"GOLD = '{ws}/eval/train/gold.json'\nOUT = '{ws}/results/x'\n"
    assert read("data/loader.py") == f"P = '{ws}/data'"
    assert read("data/run") == f"#!/bin/bash\ncd {ws}/work\n"          # a script without a suffix
    assert read("data/notes") == "see /home/user/work\n"                 # not a script: content
    assert read("method/config.json") == f'{{"out": "{ws}/outputs"}}'
    assert read("READMEs/problem.md") == f"Results go to {ws}/scoring.\n"


def test_skips_binaries_tool_state_and_downloads(tmp_path):
    ws = tmp_path / "ws"
    write(ws, {
        "eval/blob.bin": b"\0/home/user\0",
        ".git/config": "path = /home/user",
        ".venv/pyvenv.cfg": "home = /home/user",
        ".cache/x.sh": "cd /home/user",
        "trajectory/turn.jsonl": "/home/user",
        "data/downloads/readme.sh": "cd /home/user",
        "method/node_modules/pkg/index.js": "/home/user",
        "method/__pycache__/x.py": "/home/user",
        # Only the top-level directories are tool state: these are the scientist's code.
        "method/trajectory/run.sh": "cd /home/user",
        "method/downloads/get.sh": "cd /home/user",
    })
    assert localize.localize_workspace(ws) == ["method/downloads/get.sh", "method/trajectory/run.sh"]


def test_the_target_can_differ_from_where_the_files_are_now(tmp_path):
    """The workspace is rewritten before it is moved into place."""
    staging = tmp_path / ".ws.tmp-1"
    write(staging, {"scripts/run.sh": "cd /home/user\n", "data/x.csv": "/home/user/data/a\n", "data/a": "1"})
    final = tmp_path / "ws"
    assert localize.localize_workspace(staging, target=final) == ["data/x.csv", "scripts/run.sh"]
    assert (staging / "scripts" / "run.sh").read_text() == f"cd {final}\n"
    assert (staging / "data" / "x.csv").read_text() == f"{final}/data/a\n"


def test_subpaths_limit_the_rewrite(tmp_path):
    ws = tmp_path / "ws"
    write(ws, {"eval/a.py": "/home/user", "scripts/t.sh": "/home/user", "scripts/other.sh": "/home/user"})
    assert localize.localize_workspace(ws, ["eval", "scripts/t.sh", "missing"]) == ["eval/a.py", "scripts/t.sh"]
    assert (ws / "scripts" / "other.sh").read_text() == "/home/user"


def test_symlinks_are_left_alone(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside.sh"
    outside.write_text("/home/user")
    (ws / "link.sh").symlink_to(outside)
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    (outside_dir / "a.sh").write_text("/home/user")
    (ws / "linked_dir").symlink_to(outside_dir)
    assert localize.localize_workspace(ws) == []
    assert localize.localize_workspace(ws, ["linked_dir", "link.sh"]) == []
    assert outside.read_text() == "/home/user" and (outside_dir / "a.sh").read_text() == "/home/user"


@pytest.mark.parametrize("name", ["with space", "semi;colon", "dollar$x", "quote'x"])
def test_unsafe_workspace_paths_are_refused(tmp_path, name):
    ws = tmp_path / name
    ws.mkdir()
    assert not localize.is_safe_workspace_path(ws)
    with pytest.raises(ValueError):
        localize.localize_workspace(ws)


def test_command_line(tmp_path, capsys):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.sh").write_text("cd /home/user")
    assert localize.main([str(ws)]) == 0
    assert "rewrote 1 file(s)" in capsys.readouterr().out
    assert localize.main([]) == 2
    bad = tmp_path / "bad dir"
    bad.mkdir()
    assert localize.main([str(bad)]) == 1
