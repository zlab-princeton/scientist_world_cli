"""Real container runtimes: build the agent image and run a session in it.

The rest of the suite checks the commands sciworld generates. These tests run
them: ``sw build`` with Docker, Podman, Apptainer or Singularity, both agent
CLIs started in the image as the calling user, and a complete run (main phase,
then test phase) with a stand-in agent that reports what a real agent could
see and change.

They need the runtime, network access and several minutes, so they only run
when asked for::

    SCIWORLD_TEST_CONTAINERS=docker,podman pytest tests/test_container_runtimes.py -rA

``SCIWORLD_TEST_AGENT_IMAGE`` names an image that is already built (a ``.sif``
path, or a tag) and skips the build. A Docker or Podman image is built under a
tag of its own, never under the one of your real agent image, and removed
afterwards.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import sciworld
from conftest import ARXIV_ID
from sciworld import launcher as L
from sciworld.launcher import selinux_enforcing as _selinux_enforcing
from sciworld import settings as config
from sciworld import setup_cmd

RUNTIMES = [name.strip() for name in os.environ.get("SCIWORLD_TEST_CONTAINERS", "").split(",") if name.strip()]
PREBUILT = os.environ.get("SCIWORLD_TEST_AGENT_IMAGE", "")
TEST_TAG = "sciworld-agent-test:latest"
REAL_HOME = os.environ.get("HOME", "")

# Stand-ins for a login: none of these is a credential of anything.
TEST_KEYS = {"claude-code": ("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key-0000"),
             "codex": ("CODEX_API_KEY", "sk-not-a-real-key-0000")}
TEST_LOGIN_FILE = '{"note": "a stand-in for the login file of the agent"}'

pytestmark = pytest.mark.skipif(
    not RUNTIMES,
    reason="set SCIWORLD_TEST_CONTAINERS=docker,podman,apptainer to run against real container runtimes",
)

# What an agent CLI prints when it cannot look up who it runs as or read its
# own files, and when it does not know one of the options sciworld passes.
CRASH_SIGNATURES = ("uv_os_get_passwd", "getpwuid", "EACCES", "Permission denied", "permission denied")
REJECTED_OPTION = ("unknown option", "unexpected argument", "unrecognized option", "unrecognized argument")

STANDIN = r'''#!/usr/bin/env python3
"""Stand-in for the agent CLI inside the container. It reports what an agent
can see and change, runs the environment's evaluators, and prints the event
stream of the real CLI."""
import json, os, subprocess, sys
from pathlib import Path

HOST_MARKER = {marker!r}
prompt = sys.stdin.read()
home = Path("/home/user")


def emit(event):
    print(json.dumps(event), flush=True)


def can_create(directory):
    probe = Path(directory) / ".sw_probe"
    try:
        probe.write_text("x")
        probe.unlink()
        return True
    except OSError:
        return False


def can_append(path):
    try:
        with open(path, "a"):
            return True
    except OSError:
        return False


def run(script):
    result = subprocess.run(["bash", script], capture_output=True, text=True)
    return {{"status": result.returncode, "output": (result.stdout + result.stderr)[-800:]}}


test_phase = "test slice has been added" in prompt
report = {{
    "argv": sys.argv[1:],
    "uid": os.getuid(),
    "cwd": os.getcwd(),
    "home": os.environ.get("HOME"),
    "create": {{name: can_create(home / name) for name in
               ("READMEs", "eval", "baseline", "method", "scoring", "data", ".")}},
    "append": {{name: can_append(home / name) for name in
               ("scripts/evaluate_train.sh", "scripts/baseline.sh", "scripts/evaluate_test.sh",
                "scoring/scores_train.json") if (home / name).exists()}},
    "tmp": can_create("/tmp"),
    "tmpdir": can_create(os.environ.get("TMPDIR") or "/nonexistent"),
    "held_out_evaluator": (home / "eval" / "test").exists(),
    "host_marker_visible": os.path.exists(HOST_MARKER),
    "transcript_visible": (home / "agent.log").exists() or (home / "trajectory").exists(),
    "api_key": os.environ.get("ANTHROPIC_API_KEY"),
    "passed_through": os.environ.get("SW_TEST_PASSED_THROUGH"),
    "not_passed_through": os.environ.get("SW_TEST_NOT_PASSED_THROUGH"),
    "login_file": (home / ".claude" / ".credentials.json").read_text()
                  if os.access(home / ".claude" / ".credentials.json", os.R_OK) else None,
}}

emit({{"type": "system", "subtype": "init", "model": "stand-in", "cwd": os.getcwd(), "session_id": "s-1"}})
if test_phase:
    report["evaluator"] = run("scripts/evaluate_test.sh")
    (home / "reflection.md").write_text("Parity is n mod 2.\n")
    (home / "probe_test.json").write_text(json.dumps(report, indent=2))
else:
    (home / "method" / "predictions_train.json").write_text(json.dumps({{str(i): i % 2 for i in range(8)}}))
    (home / "method" / "predictions_test.json").write_text(json.dumps({{str(i): i % 2 for i in range(100, 110)}}))
    report["evaluator"] = run("scripts/evaluate_train.sh")
    (home / "idea.md").write_text("Use n mod 2.\n")
    (home / "probe_main.json").write_text(json.dumps(report, indent=2))
(home / "status.json").write_text(json.dumps({{"done": True}}))
emit({{"type": "assistant", "message": {{"content": [{{"type": "text", "text": json.dumps(report["evaluator"])}}]}}}})
emit({{"type": "result", "subtype": "success", "session_id": "s-1", "num_turns": 1,
      "total_cost_usd": 0.0, "usage": {{"input_tokens": 1, "output_tokens": 1}}}})
'''


def _apptainer(runtime: str) -> bool:
    return runtime in ("apptainer", "singularity")


@pytest.fixture(scope="module", params=RUNTIMES or ["none"])
def image(request, tmp_path_factory):
    """``(runtime, image)``: the agent image, built once per runtime with the
    code behind ``sw build``."""
    runtime = request.param
    assert runtime in L.SUPPORTED_RUNTIMES, f"unknown runtime in SCIWORLD_TEST_CONTAINERS: {runtime}"
    assert shutil.which(runtime), f"SCIWORLD_TEST_CONTAINERS asks for {runtime}, which is not installed"

    patch = pytest.MonkeyPatch()
    # Rootless Podman keeps its images under HOME, so the build and the runs
    # share one throwaway HOME. The Docker client still needs its own
    # configuration to find the daemon.
    home = tmp_path_factory.mktemp("home")
    patch.setenv("HOME", str(home))
    docker_config = Path(REAL_HOME) / ".docker"
    if REAL_HOME and docker_config.is_dir() and not os.environ.get("DOCKER_CONFIG"):
        patch.setenv("DOCKER_CONFIG", str(docker_config))
    tag = PREBUILT if (PREBUILT and not _apptainer(runtime)) else TEST_TAG
    patch.setattr(L, "DOCKER_IMAGE_TAG", tag)
    patch.setattr(setup_cmd, "DOCKER_IMAGE_TAG", tag)

    containers = tmp_path_factory.mktemp("images")
    if PREBUILT:
        reference = PREBUILT
    else:
        assert setup_cmd.build_agent_image(containers, runtime, force=True), f"`sw build` failed with {runtime}"
        reference = L.agent_image_ref(runtime, containers)
    assert L.image_exists(runtime, reference), f"{runtime} has no image {reference}"
    try:
        yield runtime, reference, home
    finally:
        if not PREBUILT and not _apptainer(runtime):
            subprocess.run([runtime, "image", "rm", "--force", reference], capture_output=True)
        patch.undo()


@pytest.fixture(autouse=True)
def shared_home(image, isolated, monkeypatch):
    """Every test of this module runs with the HOME the image was built under,
    and with the SELinux state of this machine, which the shared fixture hides."""
    monkeypatch.setenv("HOME", str(image[2]))
    monkeypatch.setattr(L, "selinux_enforcing", _selinux_enforcing)


@pytest.fixture
def container_setup(image, shared_home, data_dir, tmp_path, monkeypatch):
    """sciworld configured for the runtime, with the stand-in agent exposed
    in the container in place of ``claude``, and with a login on the host:
    a login file, an API key and one variable the user chose to pass through."""
    runtime, reference, home = image
    (home / ".claude").mkdir(exist_ok=True)
    (home / ".claude" / ".credentials.json").write_text(TEST_LOGIN_FILE)
    monkeypatch.setenv(*TEST_KEYS["claude-code"])
    monkeypatch.setenv("SW_TEST_PASSED_THROUGH", "a value with spaces and 'quotes'")
    monkeypatch.setenv("SW_TEST_NOT_PASSED_THROUGH", "stays on the host")
    if _apptainer(runtime):
        (data_dir / "containers" / "agent.sif").symlink_to(reference)
    marker = home / "host_marker.txt"
    marker.write_text("a file of the host\n")
    standin = tmp_path / "standin" / "claude"
    standin.parent.mkdir()
    standin.write_text(STANDIN.format(marker=str(marker)))
    standin.chmod(0o755)
    cfg = dict(config.DEFAULT_CONFIG)
    cfg.update(data_dir=str(data_dir), container=runtime, session="nohup", bind_paths=[str(standin)],
               passthrough_env=["SW_TEST_PASSED_THROUGH"])
    config.save_config(cfg)
    return runtime, reference


def _in_container(runtime: str, reference: str, workspace: Path, cmd: list, stdin: str = "") -> subprocess.CompletedProcess:
    argv = L.build_container_cmd(runtime, reference, workspace, cmd=cmd)
    return subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=600)


def test_both_agent_clis_start_in_the_image_as_the_calling_user(image, tmp_path):
    runtime, reference, _ = image
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for binary in ("claude", "codex"):
        result = _in_container(runtime, reference, workspace, [binary, "--version"])
        print(f"[{runtime}] {binary} --version -> {result.returncode}: {(result.stdout + result.stderr).strip()[:300]}")
        assert result.returncode == 0, result.stdout + result.stderr
        assert re.search(r"\d+\.\d+", result.stdout), result.stdout

    whoami = _in_container(runtime, reference, workspace, ["sh", "-c", "id -u; pwd; echo $HOME; touch made_inside"])
    assert whoami.returncode == 0, whoami.stderr
    assert whoami.stdout.split() == [str(os.getuid()), "/home/user", "/home/user"]
    assert (workspace / "made_inside").stat().st_uid == os.getuid()


def test_the_agent_clis_get_as_far_as_asking_for_credentials(image, tmp_path):
    """Started exactly as a run starts them, but without a login, the CLIs
    must fail for that reason: not because they reject an option, and not
    because the container user has no account entry or cannot read its files."""
    runtime, reference, _ = image
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for agent in ("claude-code", "codex"):
        result = _in_container(runtime, reference, workspace, L.build_agent_cmd(agent), stdin="Say hello.\n")
        output = result.stdout + result.stderr
        print(f"[{runtime}] {agent} without credentials -> {result.returncode}:\n{output.strip()[-1200:]}\n")
        assert result.returncode != 0, f"{agent} ran without credentials under {runtime}"
        for signature in CRASH_SIGNATURES + REJECTED_OPTION:
            assert signature not in output, f"{agent} under {runtime}: {output[-1200:]}"


def test_a_key_given_to_the_container_reaches_the_agent_cli(image, tmp_path):
    """With a key that is not one, the CLIs must get one step further than
    without: the provider turns the key down, the CLI no longer asks for a login."""
    runtime, reference, _ = image
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    asks_for_login = {"claude-code": "Not logged in", "codex": "Missing bearer"}
    for agent, (variable, value) in TEST_KEYS.items():
        argv = L.build_container_cmd(runtime, reference, workspace, env={variable: value},
                                     cmd=L.build_agent_cmd(agent))
        result = subprocess.run(argv, input="Say hello.\n", capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr
        print(f"[{runtime}] {agent} with a key that is not one -> {result.returncode}:\n{output.strip()[-900:]}\n")
        assert result.returncode != 0
        assert asks_for_login[agent] not in output, f"{agent} under {runtime} did not see {variable}"
        for signature in CRASH_SIGNATURES + REJECTED_OPTION:
            assert signature not in output, f"{agent} under {runtime}: {output[-1200:]}"


def test_a_complete_run_inside_the_container(container_setup):
    runtime, _ = container_setup
    run = sciworld.research(id=ARXIV_ID, agent="claude-code", mode="improve", no_launch=True)
    workspace = run["env_dir"]
    done = subprocess.run(
        ["bash", str(run["launch_script"])], capture_output=True, text=True, timeout=1200,
        env=dict(os.environ, SCIWORLD_RETRY_DELAY="0"),
    )
    log = run["log"].read_text() if run["log"].exists() else ""
    assert done.returncode == 0, f"{done.stdout[-2500:]}\n{done.stderr[-1500:]}\n{log[-2500:]}"

    main = json.loads((workspace / "probe_main.json").read_text())
    test = json.loads((workspace / "probe_test.json").read_text())
    print(f"[{runtime}] main phase: {json.dumps(main)}\n[{runtime}] test phase: {json.dumps(test)}")

    # Who and where the agent is.
    for phase in (main, test):
        assert (phase["uid"], phase["cwd"], phase["home"]) == (os.getuid(), "/home/user", "/home/user")
        assert phase["tmp"] and phase["tmpdir"]
        assert not phase["host_marker_visible"] and not phase["transcript_visible"]
        assert phase["evaluator"]["status"] == 0, phase["evaluator"]["output"]
        # The login of the host reached the agent: the file, the key, and the
        # variable the user passes through. Nothing else of the host did.
        assert phase["login_file"] == TEST_LOGIN_FILE
        assert phase["api_key"] == TEST_KEYS["claude-code"][1]
        assert phase["passed_through"] == "a value with spaces and 'quotes'"
        assert phase["not_passed_through"] is None
    assert "--continue" not in main["argv"] and test["argv"][0] == "--continue"
    # On the host the copies are readable by their owner only, and the key
    # is in the private file of the run, not in the launcher.
    records = run["launch_script"].parent
    assert (workspace / ".claude" / ".credentials.json").stat().st_mode & 0o777 == 0o600
    assert (records / "launch.env").stat().st_mode & 0o777 == 0o600
    assert TEST_KEYS["claude-code"][1] in (records / "launch.env").read_text()
    assert TEST_KEYS["claude-code"][1] not in run["launch_script"].read_text()

    # Main phase: the task, the evaluator and the baseline are read-only;
    # the method, the scores and the data are the agent's to write.
    assert main["create"] == {"READMEs": False, "eval": False, "baseline": False,
                              "method": True, "scoring": True, "data": True, ".": True}
    assert main["append"] == {"scripts/evaluate_train.sh": False, "scripts/baseline.sh": False}
    assert not main["held_out_evaluator"]

    # Test phase: the held-out evaluator is there, and the method and the
    # visible scores can no longer change.
    assert test["held_out_evaluator"]
    assert test["create"] == {"READMEs": False, "eval": False, "baseline": False,
                              "method": False, "scoring": True, "data": True, ".": True}
    assert test["append"] == {"scripts/evaluate_train.sh": False, "scripts/baseline.sh": False,
                              "scripts/evaluate_test.sh": False, "scoring/scores_train.json": False}

    # What the run left behind belongs to the calling user and scores.
    for name in ("probe_main.json", "probe_test.json", "idea.md", "reflection.md", "status.json",
                 "scoring/scores_train.json", "scoring/scores_test.json", "method/predictions_train.json"):
        assert (workspace / name).stat().st_uid == os.getuid(), name
    for split, count in (("train", 8), ("test", 10)):
        scores = json.loads((workspace / "scoring" / f"scores_{split}.json").read_text())
        assert scores["experiments"]["main"]["results"]["ours"] == {"accuracy": 1.0, "n": count}
    result = sciworld.evaluate(id=ARXIV_ID, run=run["run_name"])
    assert result["splits"]["train"]["overall"] == pytest.approx(1.0)
