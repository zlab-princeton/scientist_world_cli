"""The `sw` command line: parsing, the non-interactive commands, and exit codes."""

import json
import re
from pathlib import Path

import pytest

import sciworld
from conftest import ARXIV_ID
from sciworld import cli, settings


def run(*argv):
    return cli.main(list(argv))


# ── Parser ──

def test_version_matches_the_package_metadata(capsys):
    with pytest.raises(SystemExit) as exit_info:
        run("--version")
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"sciworld {sciworld.__version__}"
    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    assert re.search(r'^version = "(.+)"$', pyproject, re.M).group(1) == sciworld.__version__


def test_no_command_prints_help(capsys):
    assert run() == 0
    out = capsys.readouterr().out
    for command in ("init", "config", "build", "list", "select", "info", "research", "status", "evaluate"):
        assert command in out


@pytest.mark.parametrize("command", ["init", "config", "build", "list", "select", "info",
                                     "research", "status", "evaluate", "eval"])
def test_every_command_has_help(command, capsys):
    with pytest.raises(SystemExit) as exit_info:
        run(command, "--help")
    assert exit_info.value.code == 0
    assert "usage: sw" in capsys.readouterr().out


def parse(*argv):
    args = cli.build_parser().parse_args(list(argv))
    for dest in ("paper_id", "arxiv_id"):
        if hasattr(args, dest):
            cli._merge_paper(args, dest)
    return args


def test_paper_is_positional_and_the_old_flag_still_works():
    assert parse("research", "2502.13759").paper_id == "2502.13759"
    assert parse("research", "--id", "2502.13759").paper_id == "2502.13759"
    assert parse("research").paper_id is None
    assert parse("evaluate", "2502.13759", "--run", "r").arxiv_id == "2502.13759"
    assert parse("eval", "--id", "2502.13759").arxiv_id == "2502.13759"
    assert parse("info", "2502.13759", "--full").full is True
    assert parse("select", "2502.13759").paper_id == "2502.13759"
    assert parse("status").arxiv_id is None


def test_research_options():
    args = parse("research", "2502.13759", "--mode", "scratch", "--agent", "codex", "--model", "m",
                 "--env", "podman", "--tag", "v2", "--loop", "--max-loops", "4", "--no-launch", "--overwrite")
    assert (args.mode, args.agent, args.model, args.env, args.tag) == ("scratch", "codex", "m", "podman", "v2")
    assert args.loop and args.max_loops == 4 and args.no_launch and args.overwrite
    assert parse("research", "--launcher").launcher is True  # old spelling of --no-launch
    defaults = parse("research")
    assert defaults.max_loops == 10 and not defaults.setup_only and defaults.env is None


@pytest.mark.parametrize("argv", [
    ("research", "--mode", "reproduce"), ("research", "--agent", "gemini"),
    ("research", "--env", "lxc"), ("init", "--backend", "lxc"), ("init", "--session", "zellij"),
    ("list", "--min-milestone", "perfect"), ("config", "nonsense"), ("nonsense",),
    ("list", "--limit", "0"), ("list", "--limit", "-1"), ("list", "--limit", "many"),
    ("evaluate", "--mode", "reproduce"),
])
def test_invalid_choices_are_rejected(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.build_parser().parse_args(list(argv))
    assert exit_info.value.code == 2


def test_global_org_is_not_overwritten_by_a_subcommand_default():
    assert parse("--org", "MyOrg", "list").org == "MyOrg"
    assert parse("--org", "MyOrg", "init", "--yes").org == "MyOrg"
    assert parse("init", "--org", "Other").org == "Other"
    assert parse("list").org is None


@pytest.mark.parametrize("command", ["init", "config", "build", "list", "select", "info",
                                     "research", "status", "evaluate"])
def test_org_and_verbose_are_accepted_before_and_after_the_command(command):
    assert parse(command, "--org", "After").org == "After"
    assert parse("--org", "Before", command).org == "Before"
    assert parse("--org", "Before", command, "--org", "After").org == "After"
    assert parse(command, "-v").verbose is True and parse("-v", command).verbose is True
    assert parse(command).verbose is False


# ── init / config ──

def test_scripted_init_writes_the_settings(tmp_path, capsys):
    data = tmp_path / "data"
    assert run("init", "--data-dir", str(data), "--backend", "venv", "--session", "nohup",
               "--compute", "Run everything locally.") == 0
    cfg = settings.load_config()
    assert (cfg["data_dir"], cfg["container"], cfg["session"]) == (str(data), "venv", "nohup")
    assert (data / "envs").is_dir() and (data / "containers").is_dir()
    assert settings.load_instruction("compute") == "Run everything locally."
    # Every prompt stays with the package, so upgrades reach the user.
    for name in ("system_prompt", "mode_improve", "mode_innovate", "continue", "test"):
        assert not settings.has_instruction(name), name
    from sciworld import research_cmd
    assert research_cmd.system_prompt().startswith("You are an autonomous AI researcher")
    assert "default" in capsys.readouterr().out


def test_init_build_is_scripted_and_reports_a_failed_build(tmp_path, monkeypatch, capsys):
    """`sw init --build` alone used to start the wizard, and a failed build
    still exited 0."""
    from sciworld import setup_cmd
    assert run("init", "--data-dir", str(tmp_path / "d"), "--build") == 0   # venv: nothing to build
    assert "needs no image" in capsys.readouterr().out
    built = []

    def fake_build(containers, runtime, ask=False, force=False):
        built.append((runtime, ask))
        return False
    monkeypatch.setattr(setup_cmd, "build_agent_image", fake_build)
    monkeypatch.setattr(setup_cmd.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert run("init", "--backend", "apptainer", "--session", "nohup", "--build") == 1
    assert built == [("apptainer", False)] and "could not be built" in capsys.readouterr().out
    assert settings.load_config()["container"] == "apptainer"   # the settings were saved all the same
    # On a terminal --build still asks nothing.
    from sciworld import style
    monkeypatch.setattr(style, "is_interactive", lambda: True)
    monkeypatch.setattr(setup_cmd, "build_agent_image", lambda *a, **k: built.append("again") or True)
    assert run("init", "--build") == 0 and built[-1] == "again"


def test_scripted_init_needs_a_data_directory(capsys):
    assert run("init", "--yes") == 1
    assert "data directory is required" in capsys.readouterr().out
    assert not settings.is_initialized()


def test_scripted_init_keeps_what_is_already_set(tmp_path):
    assert run("init", "--data-dir", str(tmp_path / "a"), "--backend", "venv", "--session", "nohup") == 0
    assert run("init", "--session", "nohup", "--yes") == 0
    assert settings.load_config()["data_dir"] == str(tmp_path / "a")


def test_init_without_a_terminal_is_scripted(tmp_path):
    # stdin is not a terminal under pytest: the wizard must not start.
    assert run("init") == 1
    assert run("init", "--data-dir", str(tmp_path / "d")) == 0
    assert settings.load_config()["container"] == "venv"


def test_init_does_not_persist_environment_overrides(tmp_path, monkeypatch):
    assert run("init", "--data-dir", str(tmp_path / "real"), "--session", "nohup") == 0
    monkeypatch.setenv("SCIWORLD_DATA_DIR", str(tmp_path / "temporary"))
    assert run("init", "--backend", "venv") == 0
    assert settings.load_config(apply_env=False)["data_dir"] == str(tmp_path / "real")


def test_config_list_path_and_reset(configured, capsys):
    assert run("config", "path") == 0
    assert capsys.readouterr().out.strip() == str(settings.config_dir())
    assert run("config", "list") == 0
    out = capsys.readouterr().out
    assert configured["data_dir"] in out and "venv" in out and "task (innovate)" in out
    settings.save_instruction("compute", "x")
    assert run("config", "reset") == 1          # no terminal and no --yes
    assert settings.is_initialized()
    assert run("config", "reset", "--yes") == 0
    assert not settings.is_initialized() and settings.load_instruction("compute") == ""
    assert Path(configured["data_dir"]).is_dir()  # data is never touched
    assert run("config", "reset", "--yes") == 0   # nothing left: still fine


def test_config_sections_need_a_terminal(configured, capsys):
    assert run("config", "session") == 1
    assert "needs a terminal" in capsys.readouterr().out


def test_config_before_init(capsys):
    assert run("config", "paths") == 1
    assert "sw init" in capsys.readouterr().out
    assert run("config", "list") == 0  # listing the defaults is harmless


def test_unreadable_settings_fail_cleanly(capsys):
    settings.config_dir().mkdir(parents=True)
    settings.config_file().write_text("{broken")
    for argv in (("config", "list"), ("list",), ("status",), ("research", ARXIV_ID), ("evaluate", ARXIV_ID)):
        assert run(*argv) == 1, argv
    assert "cannot read" in capsys.readouterr().out


def test_build_without_a_runtime(configured, capsys):
    assert run("build") == 0  # venv: nothing to build
    assert "needs no image" in capsys.readouterr().out
    settings.save_config(dict(settings.load_config(), container=""))
    assert run("build") == 1


# ── list / status / info / evaluate ──

PAPERS = [
    {"arxiv_id": "2501.00001", "milestone": "core_claim", "title": "Graph Networks", "topics": ["done", "area-graphs"],
     "repo_name": "2501.00001-g", "clone_url": "u", "html_url": "h", "private": True},
    {"arxiv_id": "2502.00002", "milestone": "full", "title": "Language Models", "topics": ["done", "area-nlp"],
     "repo_name": "2502.00002-l", "clone_url": "u", "html_url": "h", "private": False},
    {"arxiv_id": "2503.00003", "milestone": "method_runs", "title": "Vision", "topics": ["done"],
     "repo_name": "2503.00003-v", "clone_url": "u", "html_url": "h", "private": True},
]


@pytest.fixture
def listing(monkeypatch):
    from sciworld import papers
    calls = []

    def fake_list(org, token=None, on_progress=None, include_unfinished=False):
        from sciworld.github import RepoInfo
        calls.append(org)
        return [RepoInfo.from_dict(p) for p in PAPERS]

    monkeypatch.setattr(papers, "list_environments", fake_list)
    return calls


def test_list_json_filters_and_sorts(listing, capsys):
    assert run("list", "--json") == 0
    ids = [p["arxiv_id"] for p in json.loads(capsys.readouterr().out)]
    assert ids == ["2502.00002", "2501.00001", "2503.00003"]  # best reproduced first
    assert run("list", "--json", "--min-milestone", "core_claim", "--search", "graph") == 0
    assert [p["arxiv_id"] for p in json.loads(capsys.readouterr().out)] == ["2501.00001"]
    assert run("list", "--json", "--limit", "1") == 0
    assert len(json.loads(capsys.readouterr().out)) == 1


def test_list_is_cached_until_refreshed(listing, capsys):
    assert run("list") == 0
    assert run("--org", "Other", "list") == 0
    assert run("list") == 0
    assert listing == ["ScientistWorld", "Other"]
    assert run("list", "--refresh") == 0
    assert listing == ["ScientistWorld", "Other", "ScientistWorld"]
    out = capsys.readouterr().out
    assert "Language Models" in out and "3 environments" in out


def test_list_failure(monkeypatch, capsys):
    from sciworld import papers
    from sciworld.github import GitHubError

    def fail(*args, **kwargs):
        raise GitHubError("Listing environments of X: GitHub API rate limit reached.")
    monkeypatch.setattr(papers, "list_environments", fail)
    assert run("list") == 1
    assert "rate limit" in capsys.readouterr().out
    assert run("list", "--json") == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_python_api_listing(listing):
    found = sciworld.list_environments(search="language")
    assert [p["arxiv_id"] for p in found] == ["2502.00002"]


def test_status_and_info(configured, capsys):
    assert run("status") == 0
    assert "No runs" in capsys.readouterr().out
    workspace = sciworld.research(id=ARXIV_ID, agent="codex", mode="improve", setup_only=True)["env_dir"]
    capsys.readouterr()
    assert run("status", "--json") == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1 and rows[0]["run"] == "codex-improve" and rows[0]["state"] == "not started"
    assert rows[0]["scores_train"] is False and rows[0]["path"] == str(workspace)
    (workspace / "status.json").write_text('{"done": true}')
    (workspace / "agent.log").write_text("log")
    (workspace / "scoring" / "scores_train.json").write_text("{}")
    assert run("status", ARXIV_ID) == 0
    assert "finished" in capsys.readouterr().out
    assert run("status", "2499.99999", "--json") == 0
    assert json.loads(capsys.readouterr().out) == []

    assert run("info", ARXIV_ID) == 0
    out = capsys.readouterr().out
    assert "Parity Is All You Need" in out and "codex-improve" in out and "main" in out
    assert run("info", ARXIV_ID, "--full") == 0
    assert "lookup table" in capsys.readouterr().out
    assert run("info", "2499.99999") == 1
    assert "sw select 2499.99999" in capsys.readouterr().out
    assert run("info") == 1


def test_evaluate_command(configured, capsys, monkeypatch):
    from sciworld import normscore
    from sciworld.github import GitHubError

    def offline(*args, **kwargs):
        raise GitHubError("cannot reach api.github.com")
    monkeypatch.setattr(normscore, "find_environment", offline)

    assert run("evaluate", ARXIV_ID) == 1
    assert "No runs found" in capsys.readouterr().out
    workspace = sciworld.research(id=ARXIV_ID, agent="codex", mode="improve", setup_only=True)["env_dir"]
    assert run("evaluate", ARXIV_ID) == 1
    assert "No scores to compare" in capsys.readouterr().out

    scores = {"experiments": {"main": {"results": {"paper_method": {"accuracy": 0.75}}}}}
    (workspace / "scoring" / "scores_train.json").write_text(json.dumps(scores))
    (workspace / "idea.md").write_text("Use a better table.\n")
    assert run("evaluate", ARXIV_ID) == 0
    out = capsys.readouterr().out
    assert "Visible split" in out and "+50.00%" in out and "higher is better" in out
    assert "Use a better table." in out and "Held-out split" not in out
    assert "FNI" in out and "cannot reach api.github.com" in out

    assert run("evaluate", ARXIV_ID, "--run", "nope") == 1
    assert "Available: codex-improve" in capsys.readouterr().out
    assert run("evaluate", "--path", str(workspace)) == 0
    assert run("evaluate", "--path", str(workspace), "--mode", "innovate", "--org", "Elsewhere") == 0
    assert run("evaluate", "--path", str(workspace / "missing")) == 1
    assert run("evaluate") == 1

    # A score file that is not text must not crash the command.
    (workspace / "scoring" / "scores_test.json").write_bytes(b"\xff\xfe\x00")
    assert run("evaluate", ARXIV_ID) == 0
    capsys.readouterr()

    sciworld.research(id=ARXIV_ID, agent="codex", mode="innovate", setup_only=True)
    capsys.readouterr()
    assert run("evaluate", ARXIV_ID) == 1  # two runs, no terminal to ask on
    assert "Pass --run" in capsys.readouterr().out


# ── sw build ──

def _fake_runtime(tmp_path, monkeypatch, body):
    """A stand-in `apptainer` that logs its arguments and runs *body*."""
    bindir = tmp_path / "runtime-bin"
    bindir.mkdir(exist_ok=True)
    script = bindir / "apptainer"
    script.write_text(f'#!/bin/bash\necho "$@" >> "{tmp_path}/runtime.log"\n{body}\n')
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{__import__('os').environ['PATH']}")
    return tmp_path / "runtime.log"


def test_build_runs_the_runtime_and_reports_the_outcome(configured, tmp_path, monkeypatch, capsys):
    settings.save_config(dict(settings.load_config(), container="apptainer"))
    image = Path(configured["data_dir"]) / "containers" / "agent.sif"
    # `apptainer build --force IMAGE RECIPE`: the stand-in writes the image.
    log = _fake_runtime(tmp_path, monkeypatch, 'if [ "$1" = build ]; then echo image > "$3"; fi')
    assert run("build") == 0
    assert image.read_text() == "image\n"
    built = log.read_text().split()
    assert built[:3] == ["build", "--force", str(image)] and built[3].endswith("sciworld/containers/agent.def")
    assert Path(built[3]).is_file()
    assert "Built" in capsys.readouterr().out

    assert run("build") == 0                     # already there: nothing runs
    assert "already built" in capsys.readouterr().out and len(log.read_text().splitlines()) == 1
    assert run("build", "--force") == 0          # unless forced
    assert len(log.read_text().splitlines()) == 2

    _fake_runtime(tmp_path, monkeypatch, 'echo "FATAL: no network" >&2; exit 255')
    assert run("build", "--force") == 1
    out = capsys.readouterr().out
    assert "Build failed" in out and "no network" in out and "APPTAINER_TMPDIR" not in out

    # A build directory that is too small is the usual reason on a cluster:
    # say which variable moves it.
    _fake_runtime(tmp_path, monkeypatch, 'echo "FATAL: while creating SIF: write agent.sif: disk quota exceeded" >&2; exit 255')
    assert run("build", "--force") == 1
    out = capsys.readouterr().out
    assert "disk quota exceeded" in out and "Set APPTAINER_TMPDIR to such a directory" in out

    # A runtime that is not installed: reported, and nothing is started.
    from sciworld import launcher, setup_cmd
    real_which = setup_cmd.shutil.which
    monkeypatch.setattr(setup_cmd.shutil, "which", lambda name: None if name == "docker" else real_which(name))
    monkeypatch.setattr(launcher, "image_exists", lambda runtime, ref: False)
    monkeypatch.setattr(setup_cmd, "image_exists", lambda runtime, ref: False)
    assert run("build", "--runtime", "docker") == 1
    assert "docker not found in PATH" in capsys.readouterr().out
