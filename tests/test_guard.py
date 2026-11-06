"""Host-side steps on a workspace the agent has had its hands on."""

import json
import os

import pytest

from sciworld import guard


def test_mounts_lists_what_exists_in_the_given_order(tmp_path):
    ws = tmp_path / "ws"
    (ws / "eval").mkdir(parents=True)
    (ws / "scripts").mkdir()
    (ws / "scripts" / "baseline.sh").write_text("")
    assert guard.existing_mounts(ws, ["READMEs", "scripts/baseline.sh", "eval"]) == ["scripts/baseline.sh", "eval"]


def test_mounts_refuses_a_path_that_leads_out_of_the_workspace(tmp_path):
    ws = tmp_path / "ws"
    (ws / "scripts").mkdir(parents=True)
    host = tmp_path / "host"
    host.mkdir()
    (host / "baseline.sh").write_text("")
    (ws / "method").symlink_to(host)                         # the path itself
    with pytest.raises(ValueError, match="method .* leads outside"):
        guard.existing_mounts(ws, ["method"])
    (ws / "method").unlink()
    os.rename(ws / "scripts", ws / "old_scripts")
    (ws / "scripts").symlink_to(host)                        # a directory on the way
    with pytest.raises(ValueError, match="scripts/baseline.sh"):
        guard.existing_mounts(ws, ["scripts/baseline.sh"])


def test_mounts_accepts_links_that_stay_inside_and_ignores_broken_ones(tmp_path):
    ws = tmp_path / "ws"
    (ws / "v2").mkdir(parents=True)
    (ws / "method").symlink_to("v2")
    (ws / "baseline").symlink_to("/home/user/v2")  # valid in the container only: broken here
    assert guard.existing_mounts(ws, ["method", "baseline"]) == ["method"]


def test_prepare_replaces_a_link_at_the_cache_directory(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    host = tmp_path / "host"
    host.mkdir()
    (ws / ".cache").symlink_to(host)
    guard.prepare_workspace(ws)
    assert (ws / ".cache" / "tmp").is_dir() and not (ws / ".cache").is_symlink()
    assert list(host.iterdir()) == []


def _source(tmp_path):
    source = tmp_path / "source"
    (source / "eval" / "test").mkdir(parents=True)
    (source / "eval" / "train").mkdir()
    (source / "eval" / "test" / "labels.json").write_text("{}")
    (source / "eval" / "common.py").write_text("X = '/home/user/eval'\n")
    (source / "scripts").mkdir()
    (source / "scripts" / "evaluate_test.sh").write_text("#!/bin/bash\ncd /home/user\n")
    return source


def test_install_test_slice(tmp_path):
    ws = tmp_path / "ws"
    (ws / "eval" / "train").mkdir(parents=True)
    (ws / "eval" / "tampered.py").write_text("agent was here\n")
    (ws / "status.json").write_text('{"done": true}')
    (ws / "scoring").mkdir()
    (ws / "scoring" / "scores_train.json").write_text("{}")
    (ws / "scoring" / "scores_test.json").write_text('{"made": "up before the evaluator existed"}')
    guard.install_test_slice(ws, _source(tmp_path))
    assert (ws / "eval" / "test" / "labels.json").exists()
    assert not (ws / "eval" / "tampered.py").exists()        # eval/ is replaced, not merged
    script = ws / "scripts" / "evaluate_test.sh"
    assert os.access(script, os.X_OK) and "cd /home/user" in script.read_text()
    # The status file is the scientist's: an agent that set it and later
    # found it changed back did not notice, and left a finished run unfinished.
    assert json.loads((ws / "status.json").read_text()) == {"done": True}
    # Test scores older than the held-out evaluator cannot come from it.
    assert not (ws / "scoring" / "scores_test.json").exists() and not guard.has_test_scores(ws)
    assert (ws / "scoring" / "scores_train.json").exists()
    assert not list(ws.glob(".eval.tmp-*"))


def test_test_scores_count_only_as_a_file_of_the_workspace(tmp_path, capsys):
    ws = tmp_path / "ws"
    source = _source(tmp_path)
    (ws / "scoring").mkdir(parents=True)
    assert not guard.has_test_scores(ws)
    (ws / "scoring" / "scores_test.json").mkdir()               # a directory is not scores
    assert not guard.has_test_scores(ws)
    guard.install_test_slice(ws, source)                        # ... and is cleared away
    (ws / "scoring" / "scores_test.json").write_text("{}")
    assert guard.has_test_scores(ws)
    assert guard.main(["test-scores", str(ws)]) == 0 and capsys.readouterr().out.strip() == "yes"

    # A scoring/ that leads to a directory of the host: its file is neither
    # accepted as the run's scores nor removed.
    host = tmp_path / "host"
    host.mkdir()
    (host / "scores_test.json").write_text('{"mine": true}')
    other = tmp_path / "other"
    other.mkdir()
    (other / "scoring").symlink_to(host)
    assert not guard.has_test_scores(other)
    guard.install_test_slice(other, source)
    assert (host / "scores_test.json").read_text() == '{"mine": true}'
    assert guard.main(["test-scores", str(other)]) == 0 and capsys.readouterr().out.strip() == "no"


def test_install_test_slice_localizes_in_venv_mode(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    guard.install_test_slice(ws, _source(tmp_path), localize=True)
    assert f"cd {ws}" in (ws / "scripts" / "evaluate_test.sh").read_text()
    assert f"X = '{ws}/eval'" in (ws / "eval" / "common.py").read_text()


def test_install_test_slice_changes_nothing_when_the_source_has_none(tmp_path):
    ws = tmp_path / "ws"
    (ws / "eval" / "train").mkdir(parents=True)
    (ws / "status.json").write_text('{"done": true}')
    for source in (tmp_path / "missing", tmp_path / "empty"):
        source.mkdir(exist_ok=True) if source.name == "empty" else None
        with pytest.raises(FileNotFoundError):
            guard.install_test_slice(ws, source)
    assert (ws / "eval" / "train").is_dir()
    assert json.loads((ws / "status.json").read_text()) == {"done": True}


def test_install_test_slice_does_not_follow_planted_links(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    host = tmp_path / "host"
    host.mkdir()
    victim = host / "victim"
    victim.write_text("mine\n")
    (ws / "scripts").symlink_to(host)
    (ws / "eval").symlink_to(host)
    (ws / "status.json").symlink_to(victim)
    (ws / "scoring").symlink_to(host)
    guard.install_test_slice(ws, _source(tmp_path))
    assert victim.read_text() == "mine\n" and sorted(p.name for p in host.iterdir()) == ["victim"]
    assert (ws / "scripts").is_dir() and not (ws / "scripts").is_symlink()
    assert (ws / "eval" / "test").is_dir() and not (ws / "eval").is_symlink()
    # The launcher writes neither status.json nor anything under scoring/,
    # so those two are left as the agent made them.
    assert (ws / "status.json").is_symlink() and (ws / "scoring").is_symlink()


@pytest.mark.parametrize("content, expected", [
    ('{"done": true}', "yes"), ('{"done": false}', "no"), ('{"done": "true"}', "no"),
    ("[1]", "no"), ("not json", "no"), (b"\xff\xfe", "no"), (None, "no"),
])
def test_done(tmp_path, capsys, content, expected):
    if content is not None:
        (tmp_path / "status.json").write_bytes(content if isinstance(content, bytes) else content.encode())
    assert guard.main(["done", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == expected


def test_command_line(tmp_path, capsys):
    ws = tmp_path / "ws"
    (ws / "eval").mkdir(parents=True)
    assert guard.main(["mounts", str(ws), "eval", "method"]) == 0
    assert capsys.readouterr().out == "eval\n"
    (ws / "method").symlink_to(tmp_path)
    assert guard.main(["mounts", str(ws), "eval", "method"]) == guard.EXIT_ESCAPE
    captured = capsys.readouterr()
    assert captured.out == "" and "leads outside" in captured.err
    assert guard.main(["prepare", str(ws)]) == 0
    assert guard.main(["test-phase", str(ws), str(tmp_path / "nope")]) == guard.EXIT_NO_TEST_SLICE
    assert guard.main(["test-phase", str(ws), str(_source(tmp_path))]) == 0
    assert guard.main(["frobnicate"]) == 2 and guard.main([]) == 2
