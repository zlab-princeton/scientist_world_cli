"""`sw select`: downloading an environment without ever destroying anything."""

import json
from pathlib import Path

import pytest

import sciworld
from conftest import build_environment
from sciworld import cli, papers, select_cmd
from sciworld.github import GitHubError


@pytest.fixture
def github(monkeypatch):
    """A stand-in for GitHub: every paper exists, a clone builds an environment."""
    state = {"clones": [], "fail": False}

    def fake_lookup(arxiv_id, org):
        return {"arxiv_id": arxiv_id, "title": "T", "clone_url": f"https://github.com/Org/{arxiv_id}.git"}

    def fake_clone(url, dest, token=None, timeout=900):
        state["clones"].append((url, Path(dest)))
        if state["fail"]:
            raise GitHubError("git clone failed: network is unreachable")
        build_environment(Path(dest))
        (Path(dest) / "MARK").write_text(f"clone {len(state['clones'])}\n")

    monkeypatch.setattr(select_cmd, "lookup_paper", fake_lookup)
    monkeypatch.setattr(papers, "clone_repo", fake_clone)
    return state


def run(*argv):
    return cli.main(list(argv))


def test_select_downloads_into_the_data_directory(configured, github):
    source = Path(configured["data_dir"]) / "envs" / "2402.00002" / "source"
    assert run("select", "2402.00002") == 0
    assert (source / "briefing" / "problem.md").exists()
    record = json.loads((source / ".sciworld.json").read_text())
    assert record["arxiv_id"] == "2402.00002" and record["clone_url"].endswith("2402.00002.git")
    assert not list(source.parent.glob(".source.*"))  # no temporary directory left
    assert len(github["clones"]) == 1
    # The clone went to a temporary sibling first.
    assert github["clones"][0][1].parent == source.parent and github["clones"][0][1] != source


def test_an_existing_download_is_kept_unless_overwrite(configured, github, capsys):
    assert run("select", "2402.00002") == 0
    source = Path(configured["data_dir"]) / "envs" / "2402.00002" / "source"
    assert run("select", "2402.00002") == 0
    assert "Already downloaded" in capsys.readouterr().out and len(github["clones"]) == 1
    assert run("select", "2402.00002", "--overwrite") == 0
    assert (source / "MARK").read_text() == "clone 2\n"


def test_a_failed_download_keeps_the_environment_that_was_there(configured, github, capsys):
    assert run("select", "2402.00002") == 0
    source = Path(configured["data_dir"]) / "envs" / "2402.00002" / "source"
    github["fail"] = True
    assert run("select", "2402.00002", "--overwrite") == 1
    assert "network is unreachable" in capsys.readouterr().out
    assert (source / "MARK").read_text() == "clone 1\n"
    assert sorted(p.name for p in source.parent.iterdir()) == ["source"]


@pytest.mark.parametrize("overwrite", [False, True])
def test_a_directory_that_is_not_an_environment_is_never_touched(configured, github, tmp_path, capsys, overwrite):
    """`--path DIR --overwrite` used to delete DIR whatever it was."""
    target = tmp_path / "thesis"
    (target / "chapters").mkdir(parents=True)
    (target / "chapters" / "one.tex").write_text("years of work\n")
    argv = ["select", "2402.00002", "--path", str(target)] + (["--overwrite"] if overwrite else [])
    assert run(*argv) == 1
    out = capsys.readouterr().out
    assert "is not a Scientist World environment" in out and "Already downloaded" not in out
    assert (target / "chapters" / "one.tex").read_text() == "years of work\n"
    assert github["clones"] == []


def test_the_environment_of_another_paper_is_not_replaced(configured, github, tmp_path, capsys):
    target = tmp_path / "env"
    assert run("select", "2402.00002", "--path", str(target)) == 0
    assert run("select", "2403.00003", "--path", str(target), "--overwrite") == 1
    assert "holds the environment of 2402.00002" in capsys.readouterr().out
    assert json.loads((target / ".sciworld.json").read_text())["arxiv_id"] == "2402.00002"


def test_path_may_be_an_empty_or_a_new_directory(configured, github, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run("select", "2402.00002", "--path", str(empty)) == 0
    assert (empty / "briefing" / "problem.md").exists()
    assert run("select", "2402.00002", "--path", str(tmp_path / "deep" / "new")) == 0


def test_python_api_returns_the_location_and_raises_with_the_reason(configured, github, tmp_path):
    where = sciworld.select(id="2402.00002")
    assert where == Path(configured["data_dir"]) / "envs" / "2402.00002" / "source"
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.txt").write_text("x")
    with pytest.raises(RuntimeError, match="not a Scientist World environment"):
        sciworld.select(id="2402.00002", path=tmp_path / "notes", overwrite=True)


def test_unknown_paper_and_missing_arguments(configured, monkeypatch, capsys):
    monkeypatch.setattr(select_cmd, "lookup_paper", lambda arxiv_id, org: None)
    assert run("select", "2499.99999") == 1
    assert "No environment found" in capsys.readouterr().out
    assert run("select") == 1  # no terminal to browse on
    assert "No paper given" in capsys.readouterr().out
