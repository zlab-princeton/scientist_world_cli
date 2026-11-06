"""End-to-end runs in venv mode with a fake agent CLI.

Each test builds a workspace through the public API, runs the generated
launcher to completion, and checks what the agent was given and what the run
left behind. Nothing here needs a container runtime, a model, or the network.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import time
from pathlib import Path

import pytest

import sciworld
from conftest import ARXIV_ID
from sciworld import rundir


def run_launcher(script: Path, timeout: int = 300) -> subprocess.CompletedProcess:
    env = dict(os.environ, SCIWORLD_RETRY_DELAY="0")
    return subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=timeout, env=env)


def calls(record: Path) -> list:
    return json.loads(record.read_text()) if record.exists() else []


def build(**kwargs) -> dict:
    kwargs.setdefault("id", ARXIV_ID)
    kwargs.setdefault("agent", "claude-code")
    kwargs.setdefault("mode", "innovate")
    kwargs.setdefault("no_launch", True)
    return sciworld.research(**kwargs)


def test_full_run_main_then_test_phase(configured, fake_agent):
    run = build()
    workspace = run["env_dir"]
    assert workspace == Path(configured["data_dir"]) / "envs" / ARXIV_ID / "claude-code-innovate"
    assert not (workspace / "eval" / "test").exists()

    done = run_launcher(run["launch_script"])
    assert done.returncode == 0, done.stdout + done.stderr

    main, test = calls(fake_agent)
    assert main["name"] == "claude" and main["cwd"] == str(workspace)
    assert "--print" in main["argv"] and "--strict-mcp-config" in main["argv"]
    assert "--continue" not in main["argv"]
    assert "design your own method from first principles" in main["prompt"]
    assert "not in a container" in main["prompt"]
    assert main["env"]["VIRTUAL_ENV"] == str(workspace / ".venv")
    assert test["argv"][0] == "--continue"
    assert "test slice has been added" in test["prompt"]

    # The evaluator scripts are written for /home/user; they ran here because
    # the workspace was localized, including the held-out evaluator.
    train = json.loads((workspace / "scoring" / "scores_train.json").read_text())
    held_out = json.loads((workspace / "scoring" / "scores_test.json").read_text())
    assert train["experiments"]["main"]["results"]["ours"] == {"accuracy": 1.0, "n": 8}
    assert held_out["experiments"]["main"]["results"]["ours"] == {"accuracy": 1.0, "n": 10}
    assert (workspace / "eval" / "test" / "labels.json").exists()
    assert json.loads((workspace / "status.json").read_text()) == {"done": True}

    # sciworld's own files are next to the workspace, not in it.
    records = rundir.run_dir(workspace)
    assert records == workspace.parent / ".sciworld" / "claude-code-innovate"
    assert run["launch_script"] == records / "launch.sh" and run["log"] == records / "agent.log"
    trajectory = records / "trajectory"
    assert sorted(p.name for p in trajectory.glob("turn_*")) == ["turn_0001_raw.jsonl", "turn_0002_raw.jsonl"]
    stats = json.loads((trajectory / "stats.json").read_text())
    assert [t["phase"] for t in stats["turns"]] == ["main", "test"]
    assert stats["cumulative_stats"]["input_tokens"] == 20
    log = run["log"].read_text()
    assert "assistant" in log and "train: wrote" in log and "test: wrote" in log
    assert "[sciworld] test phase complete." in log
    assert not (workspace / "agent.log").exists() and not (workspace / "trajectory").exists()
    record = rundir.read_record(workspace)
    assert (record["arxiv_id"], record["mode"], record["agent"], record["backend"]) == (
        ARXIV_ID, "innovate", "claude-code", "venv")
    assert rundir.launcher_state(workspace) == {
        "alive": False, "pid": None, "pgid": None, "host": None, "exit": 0}


def test_a_run_finishes_when_the_agent_leaves_the_status_file_alone_in_the_test_phase(
        configured, fake_agent, monkeypatch):
    """A real agent ran the held-out evaluator, wrote its reflection and
    stopped: it had set done in the main phase already. The launcher used to
    reset the flag behind its back and then called the run unfinished."""
    monkeypatch.setenv("FAKE_AGENT_TEST_SETS_DONE", "0")
    run = build()
    done = run_launcher(run["launch_script"])
    assert done.returncode == 0, done.stdout + done.stderr
    assert "[sciworld] test phase complete." in run["log"].read_text()
    assert (run["env_dir"] / "scoring" / "scores_test.json").exists()
    assert json.loads((run["env_dir"] / "status.json").read_text()) == {"done": True}
    assert [row["state"] for row in status_rows()] == ["finished"]


def test_loop_resumes_until_done(configured, fake_agent, monkeypatch):
    monkeypatch.setenv("FAKE_AGENT_DONE_FIRST", "0")
    run = build(loop=True, max_loops=3)
    done = run_launcher(run["launch_script"])
    assert done.returncode == 0, done.stdout + done.stderr
    first, second, test = calls(fake_agent)
    assert "--continue" not in first["argv"]
    assert second["argv"][0] == "--continue" and "Continue your work" in second["prompt"]
    assert "test slice has been added" in test["prompt"]


def test_without_loop_an_unfinished_run_skips_the_test_phase(configured, fake_agent, monkeypatch):
    monkeypatch.setenv("FAKE_AGENT_DONE_FIRST", "0")
    run = build()
    done = run_launcher(run["launch_script"])
    # Not a success: a scheduler that runs the launcher must see a failure.
    assert done.returncode == 1
    assert len(calls(fake_agent)) == 1
    assert "skipping the test phase" in done.stdout and "did not finish" in done.stdout
    assert not (run["env_dir"] / "eval" / "test").exists()


def test_failed_first_invocation_is_restarted_not_resumed(configured, fake_agent, monkeypatch):
    monkeypatch.setenv("FAKE_AGENT_FAIL_FIRST", "1")
    run = build(loop=True, max_loops=3)
    done = run_launcher(run["launch_script"])
    assert done.returncode == 0, done.stdout + done.stderr
    crashed, restarted, test = calls(fake_agent)
    # No session exists after a crash, so the main prompt is sent again.
    assert "--continue" not in restarted["argv"]
    assert restarted["prompt"] == crashed["prompt"]
    assert "exited with status 3" in done.stdout


def test_agent_failure_is_the_launcher_exit_status(configured, fake_agent, monkeypatch):
    monkeypatch.setenv("FAKE_AGENT_FAIL_FIRST", "99")
    run = build()
    done = run_launcher(run["launch_script"])
    assert done.returncode == 3
    assert len(calls(fake_agent)) == 1
    assert json.loads((run["env_dir"] / "status.json").read_text()) == {"done": False}


def test_three_failures_in_a_row_stop_the_loop(configured, fake_agent, monkeypatch):
    monkeypatch.setenv("FAKE_AGENT_FAIL_FIRST", "99")
    run = build(loop=True, max_loops=8)
    done = run_launcher(run["launch_script"])
    assert done.returncode == 3
    assert len(calls(fake_agent)) == 3
    assert "giving up" in done.stdout


def test_codex_invocation(configured, fake_agent):
    run = build(agent="codex", mode="improve", model="some-model")
    done = run_launcher(run["launch_script"])
    assert done.returncode == 0, done.stdout + done.stderr
    main, test = calls(fake_agent)
    assert main["name"] == "codex"
    assert main["argv"][:2] == ["exec", "--json"] and main["argv"][-1] == "-"
    assert main["argv"][main["argv"].index("--model") + 1] == "some-model"
    assert test["argv"][:3] == ["exec", "resume", "--last"]
    assert "match or exceed the paper's reported scores" in main["prompt"]
    stats = json.loads((rundir.run_dir(run["env_dir"]) / "trajectory" / "stats.json").read_text())
    assert stats["scaffold"] == "codex"


def test_credentials_are_forwarded_through_a_private_file(configured, fake_agent, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-value")
    monkeypatch.setenv("MY_FORWARDED", "hello world")
    sciworld.config(passthrough_env=["MY_FORWARDED", "NOT_SET_ANYWHERE"])
    run = build()
    script = run["launch_script"]
    env_file = script.parent / "launch.env"
    assert "sk-secret-value" not in script.read_text()
    assert "sk-secret-value" in env_file.read_text()
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(script.parent.stat().st_mode) == 0o700

    # The launcher must work from a session that lacks those variables.
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("MY_FORWARDED")
    assert run_launcher(script).returncode == 0
    seen = calls(fake_agent)[0]["env"]
    assert seen["ANTHROPIC_API_KEY"] == "sk-secret-value"
    assert seen["MY_FORWARDED"] == "hello world"


def test_an_openai_key_reaches_codex_under_the_name_it_reads(configured, fake_agent, monkeypatch):
    """`codex exec` takes its key from CODEX_API_KEY and ignores OPENAI_API_KEY."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-value")
    env_file = build(agent="codex")["launch_script"].parent / "launch.env"
    assert "export OPENAI_API_KEY=sk-openai-value" in env_file.read_text()
    assert "export CODEX_API_KEY=sk-openai-value" in env_file.read_text()

    # A key given under Codex's own name is left as it is.
    monkeypatch.setenv("CODEX_API_KEY", "sk-codex-value")
    env_file = build(agent="codex", overwrite=True)["launch_script"].parent / "launch.env"
    assert "export CODEX_API_KEY=sk-codex-value" in env_file.read_text()

    # Claude Code has nothing to do with either.
    env_file = build(agent="claude-code")["launch_script"].parent / "launch.env"
    assert "CODEX_API_KEY" not in env_file.read_text() and "OPENAI_API_KEY" not in env_file.read_text()


def test_prompt_overrides_apply_to_one_call_only(configured, fake_agent):
    run = build(mode_prompt="CUSTOM TASK", system_prompt="CUSTOM SYSTEM", compute="Use one CPU.",
                test_prompt="The test slice has been added. CUSTOM TEST")
    assert run_launcher(run["launch_script"]).returncode == 0
    main, test = calls(fake_agent)
    assert main["prompt"].startswith("CUSTOM SYSTEM\n\nCUSTOM TASK\n\n## Compute\n\nUse one CPU.")
    assert test["prompt"].strip().endswith("CUSTOM TEST")
    assert sciworld.config()["data_dir"] == configured["data_dir"]
    from sciworld.settings import load_instruction
    assert load_instruction("mode_innovate") == ""


def test_launch_detached_with_nohup(configured, fake_agent):
    run = build(no_launch=False, tag="bg")
    assert run["session"] == f"sw_{ARXIV_ID.replace('.', '_')}_claude-code-innovate-bg"
    status = run["env_dir"] / "status.json"
    scores = run["env_dir"] / "scoring" / "scores_test.json"
    deadline = time.time() + 120
    while time.time() < deadline:
        if scores.exists() and json.loads(status.read_text()).get("done") is True:
            break
        time.sleep(0.5)
    else:
        pytest.fail("the detached run did not finish: " + run["log"].read_text()[-2000:])
    assert len(calls(fake_agent)) == 2
    deadline = time.time() + 30
    while rundir.launcher_state(run["env_dir"])["exit"] is None and time.time() < deadline:
        time.sleep(0.1)
    assert rundir.launcher_state(run["env_dir"])["exit"] == 0


def test_evaluate_reports_improvement_and_fni(configured, fake_agent, capsys):
    run = build()
    assert run_launcher(run["launch_script"]).returncode == 0
    coeff_dir = run["env_dir"].parent / "normalization"
    coeff_dir.mkdir()
    for split, source in (("train", 0.5), ("test", 0.6)):
        (coeff_dir / f"coeffs_innovate_{split}.json").write_text(json.dumps({
            "n_models": 4, "excluded": False, "pruned_cells": [],
            "last_updated": "2026-07-13T05:13:21.000000+00:00",
            "cells": [{"experiment": "main", "method": "paper_method", "metric": "accuracy",
                       "source": source, "higher_is_better": True, "coefficient": 2.0}],
        }))
    result = sciworld.evaluate(id=ARXIV_ID, run=run["run_name"])
    # The run calls its method "ours"; it is matched to the paper's method.
    assert result["splits"]["train"]["overall"] == pytest.approx(1.0)       # 0.5 -> 1.0
    assert result["splits"]["test"]["overall"] == pytest.approx(0.4 / 0.6)  # 0.6 -> 1.0
    assert result["fni"]["fni_train"] == pytest.approx(100.0)  # 100 * 2.0 * 0.5
    assert result["fni"]["fni_test"] == pytest.approx(80.0)    # 100 * 2.0 * 0.4
    assert result["fni"]["fni"] == pytest.approx(90.0)
    assert result["fni"]["complete"] is True
    assert result["mode"] == "innovate"
    # Which coefficients produced the number is part of the answer.
    assert result["fni"]["coefficients_updated"] == "2026-07-13"
    out = capsys.readouterr().out
    assert "reference cohort of 4 frontier models" in out and "Coefficients of 2026-07-13" in out


def test_a_launch_that_dies_at_once_is_reported(configured, fake_agent, monkeypatch, capsys):
    """`sw research` used to print "running" and exit 0 whatever happened."""
    from sciworld import launcher
    monkeypatch.setenv("FAKE_AGENT_FAIL_FIRST", "99")
    # Watch the launcher for as long as it takes this machine to fail.
    monkeypatch.setattr(launcher, "LAUNCH_WAIT_SECONDS", 120)
    monkeypatch.setattr(launcher, "LAUNCH_SETTLE_SECONDS", 120)
    with pytest.raises(RuntimeError, match="stopped right after it started with exit status 3"):
        build(no_launch=False)
    out = capsys.readouterr().out
    assert "simulated crash" in out and "is running under" not in out


def test_a_running_workspace_is_never_replaced(configured, fake_agent, monkeypatch, tmp_path):
    """--overwrite on a live run used to delete the workspace under the agent
    and start a second launcher on it."""
    gate = tmp_path / "gate"
    monkeypatch.setenv("FAKE_AGENT_WAIT_FOR", str(gate))
    run = build(no_launch=False)
    workspace = run["env_dir"]
    try:
        state = rundir.launcher_state(workspace)
        assert state["alive"] is True
        (workspace / "method" / "progress.txt").write_text("in progress\n")
        for kwargs in ({"overwrite": True}, {"overwrite": True, "setup_only": True},
                       {"overwrite": True, "no_launch": True}):
            with pytest.raises(RuntimeError, match="still active"):
                build(**kwargs)
        assert (workspace / "method" / "progress.txt").exists()
        assert rundir.launcher_state(workspace)["pid"] == state["pid"]  # still the one launcher
        rows = [r for r in status_rows() if r["run"] == workspace.name]
        assert rows[0]["state"] == "running" and rows[0]["launcher_pid"] == state["pid"]
    finally:
        gate.write_text("go")
    deadline = time.time() + 120
    while rundir.launcher_state(workspace)["alive"] and time.time() < deadline:
        time.sleep(0.2)
    assert rundir.launcher_state(workspace)["exit"] == 0
    assert [r["state"] for r in status_rows() if r["run"] == workspace.name] == ["finished"]
    build(overwrite=True)  # now it may be replaced
    assert not (workspace / "method" / "progress.txt").exists()


def status_rows():
    from sciworld import settings
    from sciworld.status_cmd import collect
    return collect(settings.load_config()["data_dir"])


def test_the_mode_of_a_run_is_recorded_not_guessed_from_its_name(configured, fake_agent):
    """A tag that contains a mode word used to select the wrong coefficients."""
    run = build(mode="improve", tag="from-scratch")
    assert run["env_dir"].name == "claude-code-improve-from-scratch"
    assert run_launcher(run["launch_script"]).returncode == 0
    coeff_dir = run["env_dir"].parent / "normalization"
    coeff_dir.mkdir()
    for mode, coefficient in (("improve", 0.5), ("innovate", 3.5)):
        for split, source in (("train", 0.5), ("test", 0.6)):
            (coeff_dir / f"coeffs_{mode}_{split}.json").write_text(json.dumps({
                "n_models": 4, "cells": [{"experiment": "main", "method": "paper_method", "metric": "accuracy",
                                          "source": source, "higher_is_better": True, "coefficient": coefficient}],
            }))
    result = sciworld.evaluate(id=ARXIV_ID, run=run["run_name"])
    assert result["mode"] == "improve"
    assert result["fni"]["fni_train"] == pytest.approx(25.0)     # 100 * 0.5 * 0.5, not 175
    # The mode can be stated, for a workspace sciworld has no record of.
    assert sciworld.evaluate(id=ARXIV_ID, run=run["run_name"], mode="innovate")["fni"]["fni_train"] == pytest.approx(175.0)


def test_a_workspace_outside_the_data_directory_is_scored_by_path(configured, fake_agent, tmp_path, capsys):
    from conftest import build_environment
    build_environment(tmp_path / "elsewhere" / "source")
    run = build(path=tmp_path / "elsewhere" / "trial")
    capsys.readouterr()
    assert run_launcher(run["launch_script"]).returncode == 0
    # The hint the launcher printed works as it stands.
    hint = [l for l in run["log"].read_text().splitlines() if "score it with" in l][0]
    assert hint.endswith(f"sw evaluate {ARXIV_ID} --path {run['env_dir']}")
    result = sciworld.evaluate(path=run["env_dir"])     # paper and mode come from the run's record
    assert result["mode"] == "innovate" and result["splits"]["train"]["overall"] == pytest.approx(1.0)
    assert "unknown" not in (result["fni"].get("reason") or "")


@pytest.mark.parametrize("manager", ["tmux", "screen"])
def test_runs_under_a_session_manager_do_not_disturb_each_other(configured, fake_agent, monkeypatch, tmp_path, manager):
    """Two runs whose session names share a prefix, started through
    `sw research`: the second launch used to end the first run's session."""
    import shutil
    from sciworld import launcher
    if not shutil.which(manager):
        pytest.skip(f"{manager} is not installed")
    sciworld.config(session=manager)
    gate = tmp_path / "gate"
    monkeypatch.setenv("FAKE_AGENT_WAIT_FOR", str(gate))
    first = build(no_launch=False, tag="v2")
    second = build(no_launch=False)
    try:
        assert first["session"] == second["session"] + "-v2"
        for run in (first, second):
            assert launcher.session_exists(manager, run["session"]), run["session"]
            assert rundir.launcher_state(run["env_dir"])["alive"] is True
            assert rundir.read_record(run["env_dir"])["session_manager"] == manager
        states = {row["run"]: row["state"] for row in status_rows()}
        assert states == {"claude-code-innovate-v2": "running", "claude-code-innovate": "running"}
        # A third launch on a live workspace is refused, with the way to stop it.
        with pytest.raises(RuntimeError, match="still active.*Stop it first"):
            build(no_launch=False, overwrite=True)
    finally:
        gate.write_text("go")
    deadline = time.time() + 180
    while time.time() < deadline:
        if all(rundir.launcher_state(run["env_dir"])["exit"] is not None for run in (first, second)):
            break
        time.sleep(0.3)
    for run in (first, second):
        assert rundir.launcher_state(run["env_dir"])["exit"] == 0, run["log"].read_text()[-1500:]
        assert (run["env_dir"] / "scoring" / "scores_test.json").exists()
    states = {row["run"]: row["state"] for row in status_rows()}
    assert states == {"claude-code-innovate-v2": "finished", "claude-code-innovate": "finished"}
