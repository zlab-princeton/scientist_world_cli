"""`sw status`: what state a run is in, from what the launcher recorded."""

import json
import os
import shutil
import subprocess
import time

import pytest

from conftest import ARXIV_ID
from sciworld import cli, rundir
from sciworld.session import session_name
from sciworld.status_cmd import collect, run_status


@pytest.fixture
def run_dir(tmp_path):
    workspace = tmp_path / "envs" / ARXIV_ID / "claude-code-innovate"
    (workspace / "scoring").mkdir(parents=True)
    (workspace / "READMEs").mkdir()
    (workspace / "status.json").write_text('{"done": false}')
    return workspace


def records(workspace, **files):
    directory = rundir.ensure_run_dir(workspace)
    for name, content in files.items():
        (directory / name.replace("_", ".", 1)).write_text(content)
    return directory


def state(workspace):
    return run_status(ARXIV_ID, workspace)["state"]


def test_a_workspace_nobody_started(run_dir):
    assert state(run_dir) == "not started"
    records(run_dir)                       # built with --no-launch: records, but no launcher yet
    assert state(run_dir) == "not started"


def test_a_live_launcher_is_running_whatever_else_is_installed(run_dir):
    """A nohup or sbatch run used to show "stopped" on any machine that has
    tmux or screen installed, because no session of its name existed."""
    directory = records(run_dir, agent_log="output\n")
    script = directory / "launch.sh"
    script.write_text("#!/bin/bash\nsleep 60\n")
    process = subprocess.Popen(["bash", str(script)], start_new_session=True)
    try:
        (directory / "launcher.pid").write_text(
            f"pid={process.pid}\npgid={process.pid}\nhost={os.uname().nodename}\n")
        old = time.time() - 7200
        os.utime(directory / "agent.log", (old, old))       # silent for two hours: a long training job
        row = run_status(ARXIV_ID, run_dir)
        assert row["state"] == "running" and row["launcher_pid"] == process.pid
        (run_dir / "scripts").mkdir()
        (run_dir / "scripts" / "evaluate_test.sh").write_text("")
        assert state(run_dir) == "test phase"
    finally:
        process.kill()
        process.wait()
    assert state(run_dir) == "stopped"      # the process is gone, the record is stale


def test_a_launcher_on_another_host_is_reported_with_its_host(run_dir, capsys):
    records(run_dir, agent_log="x\n", launcher_pid="pid=4242\npgid=4242\nhost=node-17\n")
    row = run_status(ARXIV_ID, run_dir)
    assert row["state"] == "running" and row["launcher_host"] == "node-17"


def test_how_a_run_ended(run_dir):
    records(run_dir, agent_log="x\n", launcher_exit="3\n")
    row = run_status(ARXIV_ID, run_dir)
    assert row["state"] == "failed" and row["exit_status"] == 3
    records(run_dir, launcher_exit="143\n")
    assert state(run_dir) == "failed"
    records(run_dir, launcher_exit="1\n")                    # ended without done
    assert state(run_dir) == "failed"
    (run_dir / "status.json").write_text('{"done": true}')
    records(run_dir, launcher_exit="0\n")
    assert state(run_dir) == "finished"
    (run_dir / "scripts").mkdir()
    (run_dir / "scripts" / "evaluate_test.sh").write_text("")
    assert state(run_dir) == "finished (no test scores)"
    (run_dir / "scoring" / "scores_test.json").write_text("{}")
    row = run_status(ARXIV_ID, run_dir)
    assert row["state"] == "finished" and row["scores_test"] is True and row["scores_train"] is False


def test_runs_without_records_fall_back_to_the_session_and_the_log(run_dir):
    """Runs started by sciworld 0.2 kept agent.log in the workspace and left
    no launcher records."""
    (run_dir / "agent.log").write_text("x\n")
    row = run_status(ARXIV_ID, run_dir)
    assert row["log"] == str(run_dir / "agent.log") and row["state"] == "stopped"
    if not shutil.which("tmux"):
        pytest.skip("tmux is not installed")
    name = session_name(ARXIV_ID, run_dir.name)
    subprocess.run(["tmux", "new-session", "-d", "-s", name + "-v2", "sleep", "60"], check=True)
    assert state(run_dir) == "stopped"      # another run's session: an exact name is required
    subprocess.run(["tmux", "new-session", "-d", "-s", name, "sleep", "60"], check=True)
    assert state(run_dir) == "running"


@pytest.mark.parametrize("content", [b"\xff\xfe\x00", b"{broken", b"[true]", b""])
def test_an_unreadable_status_file_is_not_fatal(run_dir, content):
    (run_dir / "status.json").write_bytes(content)
    row = run_status(ARXIV_ID, run_dir)
    assert row["done"] is False and row["state"] == "not started"


def test_command_output(run_dir, tmp_path, configured, capsys, monkeypatch):
    monkeypatch.setenv("SCIWORLD_DATA_DIR", str(tmp_path))
    records(run_dir, agent_log="x\n", launcher_exit="3\n")
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "claude-code-innovate" in out and "failed (exit 3)" in out
    assert cli.main(["status", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows == collect(str(tmp_path)) and rows[0]["log"].endswith(".sciworld/claude-code-innovate/agent.log")
