"""`sw research` up to the point where the launcher is written."""

import subprocess
from pathlib import Path

import pytest

import sciworld
from conftest import ARXIV_ID, build_environment
from sciworld import research_cmd
from sciworld.github import GitHubError


def setup_only(**kwargs):
    kwargs.setdefault("id", ARXIV_ID)
    kwargs.setdefault("agent", "claude-code")
    kwargs.setdefault("mode", "innovate")
    return sciworld.research(setup_only=True, **kwargs)


@pytest.fixture(autouse=True)
def no_github(monkeypatch):
    """These tests work from a local source; reaching GitHub is an error."""
    def refuse(*args, **kwargs):
        raise AssertionError("GitHub must not be contacted")
    monkeypatch.setattr(research_cmd, "lookup_paper", refuse)
    monkeypatch.setattr(research_cmd, "clone_environment", refuse)


def test_setup_only_builds_the_workspace_and_stops(configured):
    run = setup_only()
    ws = run["env_dir"]
    assert run["run_name"] == "claude-code-innovate" and run["mode"] == "innovate"
    assert run["launch_script"] is None and run["session"] is None
    assert ws.name == "claude-code-innovate" and (ws / "READMEs" / "problem.md").exists()
    assert run["source_dir"] == ws.parent / "source"
    assert not (ws / ".venv").exists() and not (ws.parent / ".sciworld").exists()
    log = subprocess.run(["git", "log", "--format=%s"], cwd=ws, capture_output=True, text=True).stdout
    assert log.strip() == "sciworld: initial environment"


def test_old_mode_name_builds_an_innovate_run(configured):
    run = setup_only(mode="scratch", tag="v2")
    assert run["env_dir"].name == "claude-code-innovate-v2" and run["mode"] == "innovate"


def test_tag_is_made_safe_for_a_directory_name(configured):
    assert setup_only(tag=" lr sweep/1 ")["env_dir"].name == "claude-code-innovate-lr-sweep-1"
    assert setup_only(tag="  ")["env_dir"].name == "claude-code-innovate"  # blank: no tag


def test_a_tag_without_usable_characters_is_an_error(configured):
    """It used to become "no tag", which names the untagged run."""
    untagged = setup_only()["env_dir"]
    (untagged / "method" / "mine.py").write_text("work\n")
    with pytest.raises(RuntimeError, match="no usable characters"):
        setup_only(tag="!!!", overwrite=True)
    assert (untagged / "method" / "mine.py").exists()


def test_venv_workspace_uses_local_paths(configured):
    ws = setup_only(env="venv")["env_dir"]
    script = (ws / "scripts" / "evaluate_train.sh").read_text()
    assert "/home/user" not in script and f"cd {ws}" in script
    assert str(ws) in (ws / "data" / "loader.py").read_text()
    assert not list(ws.parent.glob(".claude-code-innovate.*"))  # nothing temporary left


def test_setup_only_does_not_depend_on_the_configured_backend(configured):
    """The configured backend is venv here. A harness that asks for a
    workspace gets the container's paths unless it says otherwise."""
    for env in (None, "container", "apptainer"):
        ws = setup_only(env=env, overwrite=True)["env_dir"]
        assert "cd /home/user" in (ws / "scripts" / "evaluate_train.sh").read_text(), env


def test_existing_workspace_is_protected(configured):
    ws = setup_only()["env_dir"]
    (ws / "method" / "mine.py").write_text("work\n")
    with pytest.raises(RuntimeError):
        setup_only()
    assert (ws / "method" / "mine.py").exists()
    setup_only(overwrite=True)
    assert not (ws / "method" / "mine.py").exists()


def test_a_directory_that_is_not_a_workspace_is_never_deleted(configured, tmp_path):
    target = tmp_path / "paper" / "notes"
    build_environment(tmp_path / "paper" / "source")
    target.mkdir()
    (target / "thesis.tex").write_text("years of work\n")
    with pytest.raises(RuntimeError):
        setup_only(path=target, overwrite=True)
    assert (target / "thesis.tex").read_text() == "years of work\n"


def test_explicit_path_uses_the_sibling_source(isolated, tmp_path):
    # No settings at all: --path needs no data directory.
    build_environment(tmp_path / "paper" / "source")
    run = setup_only(path=tmp_path / "paper" / "my-run", mode="improve")
    assert run["env_dir"] == tmp_path / "paper" / "my-run"
    assert (run["env_dir"] / "method" / "predict.py").exists()
    assert "cd /home/user" in (run["env_dir"] / "scripts" / "evaluate_train.sh").read_text()


def test_the_sibling_source_must_be_this_papers_environment(isolated, tmp_path):
    """With --path the environment lives in ../source. It used to be reused
    for whatever paper was asked for, and for any directory of that name."""
    import json
    source = build_environment(tmp_path / "runs" / "source")
    (source / ".sciworld.json").write_text(json.dumps({"arxiv_id": ARXIV_ID}))
    setup_only(path=tmp_path / "runs" / "a")
    with pytest.raises(RuntimeError, match=f"holds the environment of {ARXIV_ID}, not of 2499.99999"):
        setup_only(id="2499.99999", path=tmp_path / "runs" / "b")
    assert not (tmp_path / "runs" / "b").exists()

    (tmp_path / "proj" / "source").mkdir(parents=True)       # the user's own code
    (tmp_path / "proj" / "source" / "main.py").write_text("print(1)\n")
    with pytest.raises(RuntimeError, match="not a Scientist World environment"):
        setup_only(path=tmp_path / "proj" / "run1")
    assert not (tmp_path / "proj" / "run1").exists()
    with pytest.raises(RuntimeError, match="where the environment itself is kept"):
        setup_only(path=tmp_path / "runs" / "source")


def test_an_empty_target_directory_is_reused(configured, tmp_path):
    build_environment(tmp_path / "paper" / "source")
    target = tmp_path / "paper" / "run"
    target.mkdir()
    assert setup_only(path=target)["env_dir"] == target


def test_any_agent_name_is_accepted_for_setup_only(configured):
    assert setup_only(agent="kimi")["env_dir"].name == "kimi-innovate"
    with pytest.raises(RuntimeError):
        sciworld.research(id=ARXIV_ID, agent="kimi", mode="innovate", no_launch=True)


def test_missing_data_directory_is_an_error(isolated):
    with pytest.raises(RuntimeError):
        setup_only()


def test_unknown_mode_is_rejected_before_anything_is_written(configured):
    with pytest.raises(ValueError):
        setup_only(mode="reproduce")
    assert not (Path(configured["data_dir"]) / "envs" / ARXIV_ID / "claude-code-reproduce").exists()


def test_required_arguments():
    with pytest.raises(TypeError, match="agent, mode"):
        sciworld.research(id=ARXIV_ID, agent=None, mode=None)


def test_paper_is_looked_up_and_cloned_when_not_downloaded(configured, monkeypatch, tmp_path):
    cloned = []

    def fake_lookup(arxiv_id, org):
        return {"arxiv_id": arxiv_id, "title": "T", "clone_url": "https://github.com/Org/x.git"}

    def fake_clone(url, dest, arxiv_id="", replace=False):
        cloned.append((url, dest))
        build_environment(dest)

    monkeypatch.setattr(research_cmd, "lookup_paper", fake_lookup)
    monkeypatch.setattr(research_cmd, "clone_environment", fake_clone)
    run = setup_only(id="arXiv:2402.00002v3")
    assert run["arxiv_id"] == "2402.00002"
    assert cloned == [("https://github.com/Org/x.git", run["source_dir"])]


def test_unknown_paper_and_github_failures_are_reported(configured, monkeypatch, capsys):
    monkeypatch.setattr(research_cmd, "lookup_paper", lambda arxiv_id, org: None)
    with pytest.raises(RuntimeError):
        setup_only(id="2499.99999")
    assert "No environment found for" in capsys.readouterr().out

    def rate_limited(arxiv_id, org):
        raise GitHubError("GitHub API rate limit reached.")
    monkeypatch.setattr(research_cmd, "lookup_paper", rate_limited)
    with pytest.raises(RuntimeError):
        setup_only(id="2499.99999")
    assert "rate limit" in capsys.readouterr().out


def test_failed_packaging_leaves_no_partial_workspace(configured, monkeypatch):
    def explode(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(research_cmd, "package_environment", explode)
    with pytest.raises(RuntimeError, match="disk full"):
        setup_only()
    paper = Path(configured["data_dir"]) / "envs" / ARXIV_ID
    assert sorted(p.name for p in paper.iterdir()) == ["source"]


def test_a_failed_overwrite_keeps_the_workspace_that_was_there(configured, monkeypatch):
    """--overwrite used to delete first and build second."""
    ws = setup_only()["env_dir"]
    (ws / "method" / "mine.py").write_text("three days of work\n")
    real = research_cmd.package_environment

    def explode(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(research_cmd, "package_environment", explode)
    with pytest.raises(RuntimeError, match="disk full"):
        setup_only(overwrite=True)
    assert (ws / "method" / "mine.py").read_text() == "three days of work\n"
    assert sorted(p.name for p in ws.parent.iterdir()) == ["claude-code-innovate", "source"]

    monkeypatch.setattr(research_cmd, "package_environment", real)
    setup_only(overwrite=True)
    assert not (ws / "method" / "mine.py").exists()
    assert sorted(p.name for p in ws.parent.iterdir()) == ["claude-code-innovate", "source"]


def test_entries_left_out_of_the_workspace_are_reported(configured, capsys):
    source = Path(configured["data_dir"]) / "envs" / ARXIV_ID / "source"
    (source / "data" / "leak").symlink_to("/etc/hostname")
    setup_only()
    out = capsys.readouterr().out
    assert "1 entries of the environment were left out" in out and "data/leak" in out


# ── Backend resolution ──

@pytest.mark.parametrize("configured_backend, flag, expected", [
    ("venv", None, "venv"), ("docker", "venv", "venv"), ("venv", "podman", "podman"),
    ("apptainer", None, "apptainer"), ("apptainer", "container", "apptainer"),
])
def test_resolve_backend(configured_backend, flag, expected):
    assert research_cmd.resolve_backend({"container": configured_backend}, flag) == (expected, None)


def test_resolve_backend_without_configuration(monkeypatch):
    backend, error = research_cmd.resolve_backend({"container": ""}, None)
    assert backend is None and "sw init" in error
    monkeypatch.setattr(research_cmd.shutil, "which", lambda name: "/usr/bin/podman" if name == "podman" else None)
    assert research_cmd.resolve_backend({"container": ""}, "container") == ("podman", None)
    monkeypatch.setattr(research_cmd.shutil, "which", lambda name: None)
    assert research_cmd.resolve_backend({"container": "venv"}, "container")[0] is None


def test_container_run_needs_a_built_image(configured, monkeypatch, capsys):
    monkeypatch.setattr(research_cmd.shutil, "which", lambda name: f"/usr/bin/{name}")
    for _attempt in (1, 2):
        # Checked before anything is created: the second attempt must fail
        # for the same reason, not with "a workspace already exists".
        with pytest.raises(RuntimeError, match="sw build"):
            sciworld.research(id=ARXIV_ID, agent="claude-code", mode="innovate", env="apptainer", no_launch=True)
        assert "sw build" in capsys.readouterr().out
    paper = Path(configured["data_dir"]) / "envs" / ARXIV_ID
    assert sorted(p.name for p in paper.iterdir()) == ["source"]


def test_container_mode_refuses_paths_that_break_mount_specifications(configured, tmp_path, capsys):
    build_environment(tmp_path / "a:b" / "source")
    with pytest.raises(RuntimeError, match="separators"):
        sciworld.research(id=ARXIV_ID, agent="claude-code", mode="innovate", env="apptainer",
                          path=tmp_path / "a:b" / "run", no_launch=True)


def test_missing_session_manager_is_reported_before_anything_is_built(configured, fake_agent, monkeypatch):
    sciworld.config(session="tmux")
    real_which = research_cmd.shutil.which
    monkeypatch.setattr(research_cmd.shutil, "which", lambda name: None if name == "tmux" else real_which(name))
    with pytest.raises(RuntimeError, match="tmux is not installed"):
        sciworld.research(id=ARXIV_ID, agent="claude-code", mode="innovate")
    paper = Path(configured["data_dir"]) / "envs" / ARXIV_ID
    assert sorted(p.name for p in paper.iterdir()) == ["source"]


def test_venv_mode_refuses_paths_that_need_quoting(configured, tmp_path, fake_agent, capsys):
    build_environment(tmp_path / "my papers" / "source")
    with pytest.raises(RuntimeError):
        sciworld.research(id=ARXIV_ID, agent="claude-code", mode="innovate",
                          path=tmp_path / "my papers" / "run", no_launch=True)
    assert "venv mode cannot use the path" in capsys.readouterr().out


def test_venv_mode_needs_the_agent_cli_installed(configured, monkeypatch, capsys):
    monkeypatch.setattr(research_cmd.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError):
        sciworld.research(id=ARXIV_ID, agent="codex", mode="innovate", no_launch=True)
    assert "codex" in capsys.readouterr().out


# ── Prompts ──

def test_the_system_prompt_falls_back_to_the_bundled_one(isolated, monkeypatch):
    from sciworld import settings
    default = research_cmd.bundled_prompt("system_prompt")
    assert default and research_cmd.system_prompt() == default   # nothing saved, e.g. no `sw init`
    settings.save_instruction("system_prompt", "MINE")
    assert research_cmd.system_prompt() == "MINE"
    settings.save_instruction("system_prompt", "")                # set to nothing on purpose
    assert research_cmd.system_prompt() == ""
    settings.remove_instruction("system_prompt")
    assert research_cmd.system_prompt() == default


def test_a_missing_bundled_prompt_is_an_error_not_an_empty_task(monkeypatch, tmp_path):
    monkeypatch.setattr(research_cmd, "_PROMPTS_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="installation is incomplete"):
        research_cmd.bundled_prompt("innovate")


def test_the_python_api_never_asks_even_in_a_terminal(configured, monkeypatch):
    """A script calling sciworld.research() from a shell used to stop at the
    interactive "Run tag" question."""
    import sys
    from sciworld import style
    monkeypatch.setattr(style, "_IS_TTY", True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    assert style.is_interactive() is True

    def asked(*args, **kwargs):
        raise AssertionError("the Python API asked a question")
    for name in ("prompt_tag", "select", "prompt", "prompt_path", "prompt_ghost"):
        monkeypatch.setattr(style, name, asked)
    ws = setup_only()["env_dir"]
    with pytest.raises(RuntimeError, match="already exists"):
        setup_only()                                   # no "Overwrite?" menu either
    assert ws.is_dir() and style.is_interactive() is True  # restored afterwards


def test_bundled_prompts_exist_and_name_only_files_the_workspace_has():
    for name in ("improve", "innovate", "continue", "test", "system_prompt", "environment", "environment_venv"):
        assert research_cmd.bundled_prompt(name), name
    for mode in ("improve", "innovate"):
        text = research_cmd.bundled_prompt(mode)
        assert "scripts/evaluate_train.sh" in text and "scoring/scores_train.json" in text
        for stale in ("scripts/evaluate.sh", "scores.json`", "method.sh", "reproduce.sh", "run.sh"):
            assert stale not in text, (mode, stale)


def test_mode_prompt_override_and_compute_instructions(isolated):
    from sciworld import settings
    assert research_cmd.mode_prompt("scratch") == research_cmd.bundled_prompt("innovate")
    settings.save_instruction("mode_innovate", "MINE")
    assert research_cmd.mode_prompt("innovate") == "MINE"
    assert research_cmd.compute_instructions() == ""
    settings.save_instruction("compute_gpu", "srun --gres=gpu:1")
    settings.save_instruction("compute_cpu", "run locally")
    assert research_cmd.compute_instructions() == "GPU: srun --gres=gpu:1\nCPU: run locally"
    settings.save_instruction("compute", "one rule")
    assert research_cmd.compute_instructions() == "one rule"


def test_workspace_detection(tmp_path, env_source):
    from sciworld.package import package_environment
    assert research_cmd.is_workspace(package_environment(env_source, tmp_path / "ws", "improve"))
    assert not research_cmd.is_workspace(env_source)
    assert research_cmd.has_test_slice(env_source)
    assert not research_cmd.has_test_slice(tmp_path / "ws")


# ── Interactive selection ──

def test_the_paper_picker_is_used_when_no_paper_is_given_on_a_terminal(configured, monkeypatch):
    """The unit tests otherwise never reach the interactive branches."""
    from sciworld import cli, style
    listing = [{"arxiv_id": "2499.00001", "title": "Other", "clone_url": ""},
               {"arxiv_id": ARXIV_ID, "title": "Parity", "clone_url": ""}]
    monkeypatch.setattr(research_cmd, "fetch_papers", lambda org, refresh=False: listing)
    monkeypatch.setattr(style, "is_interactive", lambda: True)
    menus = []

    def fake_select(label, options, default=0):
        menus.append(label)
        return {"Select a paper": 1, "Research mode": 1, "Agent": 1}[label]

    monkeypatch.setattr(style, "select", fake_select)
    monkeypatch.setattr(style, "prompt_tag", lambda label: "trial 1")
    monkeypatch.setattr(research_cmd, "tabbed_search", lambda papers: papers[1])
    assert cli.main(["research", "--setup-only"]) == 0
    assert menus == ["Select a paper", "Research mode", "Agent"]
    workspace = Path(configured["data_dir"]) / "envs" / ARXIV_ID / "codex-innovate-trial-1"
    assert (workspace / "READMEs" / "problem.md").exists()

    # "Exit" in the picker ends the command without building anything.
    monkeypatch.setattr(style, "select", lambda label, options, default=0: 3)
    assert cli.main(["research", "--setup-only", "--mode", "improve", "--agent", "codex"]) == 0
    assert not (workspace.parent / "codex-improve").exists()


def test_select_browser_is_used_on_a_terminal(configured, monkeypatch):
    from sciworld import cli, select_cmd, style
    monkeypatch.setattr(select_cmd, "fetch_papers",
                        lambda org, refresh=False: [{"arxiv_id": ARXIV_ID, "title": "Parity", "clone_url": ""}])
    monkeypatch.setattr(style, "is_interactive", lambda: True)
    monkeypatch.setattr(style, "select", lambda label, options, default=0: 0)       # "Random", then "Keep it"
    monkeypatch.setattr(select_cmd, "dice_select", lambda papers: papers[0])
    assert cli.main(["select"]) == 0     # already downloaded by the fixture: kept


def test_the_package_has_no_undefined_names():
    """A static check: code paths that only run on a terminal are not reached
    by the other tests, so a name that no longer exists would go unnoticed."""
    import io
    api = pytest.importorskip("pyflakes.api")
    from pyflakes.reporter import Reporter
    import sciworld as package
    out, err = io.StringIO(), io.StringIO()
    api.checkRecursive([str(Path(package.__file__).parent)], Reporter(out, err))
    serious = [line for line in out.getvalue().splitlines()
               if "undefined name" in line or "referenced before assignment" in line]
    assert not serious and not err.getvalue(), serious + err.getvalue().splitlines()
