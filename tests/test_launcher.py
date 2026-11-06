import os
import shutil
import stat
import subprocess
import sys
import time

import pytest

from sciworld import launcher as L
from sciworld import rundir
# The function itself: the shared fixture replaces it on the module.
from sciworld.launcher import selinux_enforcing as _selinux_enforcing


def workspace(tmp_path):
    ws = tmp_path / "ws"
    for rel in ("READMEs", "eval/train", "baseline", "scripts", "method", "scoring"):
        (ws / rel).mkdir(parents=True)
    for rel in ("scripts/evaluate_train.sh", "scripts/baseline.sh"):
        (ws / rel).write_text("#!/bin/bash\n")
    return ws


def binds(cmd, flag):
    return [cmd[i + 1] for i, arg in enumerate(cmd) if arg == flag]


# ── Container command ──

def test_apptainer_isolation_flags(tmp_path):
    ws = workspace(tmp_path)
    cmd = L.build_container_cmd("apptainer", "/img/agent.sif", ws, cmd=["claude", "--print"])
    assert cmd[:2] == ["apptainer", "exec"]
    for flag in ("--containall", "--cleanenv", "--no-eval", "--writable-tmpfs"):
        assert flag in cmd
    assert cmd[cmd.index("--home") + 1] == f"{ws}:/home/user"
    assert cmd[cmd.index("--pwd") + 1] == "/home/user"
    assert cmd[-3:] == ["/img/agent.sif", "claude", "--print"]
    mounted = binds(cmd, "--bind")
    for rel in ("READMEs", "eval/train", "baseline", "scripts/evaluate_train.sh", "scripts/baseline.sh"):
        assert f"{ws / rel}:/home/user/{rel}:ro" in mounted
    assert "/dev/null:/usr/bin/sbatch:ro" in mounted and "/dev/null:/bin/sudo:ro" in mounted
    assert not any(":/home/user/method:ro" in m for m in mounted)
    env = binds(cmd, "--env")
    assert "PIP_USER=1" in env and "TMPDIR=/home/user/.cache/tmp" in env


def test_singularity_has_no_no_eval_flag(tmp_path):
    cmd = L.build_container_cmd("singularity", "/img/agent.sif", workspace(tmp_path))
    assert cmd[:2] == ["singularity", "exec"] and "--no-eval" not in cmd


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_docker_family(tmp_path, runtime):
    ws = workspace(tmp_path)
    cmd = L.build_container_cmd(runtime, "sciworld-agent:latest", ws, cmd=["codex", "exec"])
    assert cmd[:4] == [runtime, "run", "--rm", "--interactive"]
    assert cmd[cmd.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    volumes = binds(cmd, "--volume")
    assert f"{ws}:/home/user" in volumes and f"{ws / 'READMEs'}:/home/user/READMEs:ro" in volumes
    assert not any("/dev/null" in v for v in volumes)  # nothing to mask in an image of its own
    assert "HOME=/home/user" in binds(cmd, "--env")
    assert cmd[-3:] == ["sciworld-agent:latest", "codex", "exec"]
    # Rootless podman needs to be told to keep the caller's id; docker does not.
    assert ("--userns=keep-id" in cmd) == (runtime == "podman" and os.geteuid() != 0)


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_selinux_hosts_switch_label_separation_off_for_the_container(tmp_path, monkeypatch, runtime):
    """SELinux would hide the workspace from the container; relabeling it
    would change the user's files."""
    ws = workspace(tmp_path)
    assert "--security-opt" not in L.build_container_cmd(runtime, "image", ws)

    monkeypatch.setattr(L, "selinux_enforcing", lambda: True)
    cmd = L.build_container_cmd(runtime, "image", ws, cmd=["claude"])
    assert cmd[:6] == [runtime, "run", "--rm", "--interactive", "--security-opt", "label=disable"]
    assert cmd[-2:] == ["image", "claude"]
    # Apptainer does not confine containers with SELinux labels.
    assert "--security-opt" not in L.build_container_cmd("apptainer", "i.sif", ws)


def test_selinux_state_is_read_from_the_kernel(tmp_path):
    state = tmp_path / "enforce"
    assert _selinux_enforcing(str(state)) is False   # no SELinux: the file does not exist
    state.write_text("0")
    assert _selinux_enforcing(str(state)) is False   # permissive
    state.write_text("1\n")
    assert _selinux_enforcing(str(state)) is True


def test_only_existing_paths_are_mounted_read_only(tmp_path):
    ws = workspace(tmp_path)
    (ws / "baseline").rmdir()
    mounted = binds(L.build_container_cmd("apptainer", "i.sif", ws), "--bind")
    assert not any("/home/user/baseline" in m for m in mounted)


def test_protect_false_leaves_the_read_only_mounts_to_the_caller(tmp_path):
    ws = workspace(tmp_path)
    mounted = binds(L.build_container_cmd("apptainer", "i.sif", ws, protect=False), "--bind")
    assert not any("/home/user/" in m for m in mounted if not m.startswith("/dev/null"))


def test_a_link_out_of_the_workspace_is_never_a_mount_source(tmp_path):
    """The runtime resolves a mount source on the host. A link the agent
    planted must not make it mount a host directory into the container."""
    ws = workspace(tmp_path)
    secret = tmp_path / "host_home"
    secret.mkdir()
    shutil.rmtree(ws / "baseline")
    (ws / "baseline").symlink_to(secret)
    (ws / "scripts" / "baseline.sh").unlink()
    (ws / "scripts" / "baseline.sh").symlink_to(tmp_path / "host_home")
    mounted = binds(L.build_container_cmd("apptainer", "i.sif", ws), "--bind")
    assert not any("baseline" in m for m in mounted)
    # A link that stays inside the workspace is fine.
    (ws / "elsewhere").mkdir()
    (ws / "baseline").unlink()
    (ws / "baseline").symlink_to(ws / "elsewhere")
    mounted = binds(L.build_container_cmd("apptainer", "i.sif", ws), "--bind")
    assert f"{ws / 'baseline'}:/home/user/baseline:ro" in mounted


def test_test_phase_locks_the_method_and_the_held_out_files(tmp_path):
    ws = workspace(tmp_path)
    (ws / "eval" / "test").mkdir()
    (ws / "scripts" / "evaluate_test.sh").write_text("")
    (ws / "scoring" / "scores_train.json").write_text("{}")
    locked = binds(L.build_container_cmd("apptainer", "i.sif", ws, test_phase=True), "--bind")
    for rel in ("eval/test", "scripts/evaluate_test.sh", "method", "scoring/scores_train.json"):
        assert f"{ws / rel}:/home/user/{rel}:ro" in locked
    unlocked = binds(L.build_container_cmd("apptainer", "i.sif", ws, test_phase=True,
                                           test_phase_unlock_method=True), "--bind")
    assert f"{ws / 'method'}:/home/user/method:ro" not in unlocked
    assert f"{ws / 'eval/test'}:/home/user/eval/test:ro" in unlocked


def test_extra_binds_env_trajectory_overlay_and_extra_args(tmp_path):
    ws = workspace(tmp_path)
    tool = tmp_path / "tool"
    tool.write_text("")
    empty = tmp_path / "empty"
    empty.mkdir()
    cmd = L.build_container_cmd(
        "apptainer", "i.sif", ws, extra_binds=[(tool, "/usr/local/bin/tool"), (tmp_path / "missing", "/x")],
        env={"FOO": "bar", "SKIPPED": None}, empty_traj_overlay=empty,
        extra_args=["--nv", L.RawShellArg('"${SW_RO[@]}"')], cmd=["true"],
    )
    mounted = binds(cmd, "--bind")
    assert f"{tool}:/usr/local/bin/tool:ro" in mounted and not any(":/x:ro" in m for m in mounted)
    assert f"{empty}:/home/user/trajectory:ro" in mounted
    assert "FOO=bar" in binds(cmd, "--env") and not any(e.startswith("SKIPPED") for e in binds(cmd, "--env"))
    assert cmd[-4:] == ["--nv", '"${SW_RO[@]}"', "i.sif", "true"]


def test_a_binary_the_user_exposes_is_not_masked(tmp_path):
    ws = workspace(tmp_path)
    client = tmp_path / "sbatch"
    client.write_text("#!/bin/sh\n")
    mounted = binds(L.build_container_cmd(
        "apptainer", "i.sif", ws, extra_binds=[(client, "/usr/local/bin/sbatch")]), "--bind")
    assert f"{client}:/usr/local/bin/sbatch:ro" in mounted
    assert "/dev/null:/usr/local/bin/sbatch:ro" not in mounted
    assert "/dev/null:/usr/bin/sbatch:ro" in mounted  # the other locations stay masked


def test_workspace_paths_that_break_mount_specifications_are_rejected(tmp_path):
    for name in ("a:b", "a,b"):
        with pytest.raises(ValueError, match="separators"):
            L.build_container_cmd("apptainer", "i.sif", tmp_path / name)


def test_unknown_runtime_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported container runtime"):
        L.build_container_cmd("lxc", "i", tmp_path)
    with pytest.raises(ValueError):
        L.image_exists("lxc", "i")


def test_image_helpers(tmp_path):
    assert L.agent_image_ref("apptainer", tmp_path) == str(tmp_path / "agent.sif")
    assert L.agent_image_ref("podman", tmp_path) == "sciworld-agent:latest"
    assert L.image_exists("singularity", str(tmp_path / "agent.sif")) is False
    (tmp_path / "agent.sif").write_text("x")
    assert L.image_exists("apptainer", str(tmp_path / "agent.sif")) is True
    assert L.ro_bind_flag("docker") == "--volume" and L.ro_bind_flag("apptainer") == "--bind"
    assert L.container_env_prefix("apptainer") == "APPTAINERENV_"
    assert L.container_env_prefix("singularity") == "SINGULARITYENV_"
    assert L.container_env_prefix("docker") is None


# ── Agent and local commands ──

def test_claude_command():
    assert L.build_agent_cmd("claude-code") == [
        "claude", "--print", "--dangerously-skip-permissions", "--verbose",
        "--output-format", "stream-json", "--strict-mcp-config"]
    resumed = L.build_agent_cmd("claude-code", resume=True, model="m")
    assert resumed[:2] == ["claude", "--continue"] and resumed[-2:] == ["--model", "m"]


def test_codex_command():
    assert L.build_agent_cmd("codex") == [
        "codex", "exec", "--json", "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check", "-"]
    resumed = L.build_agent_cmd("codex", resume=True, model="m")
    assert resumed[:4] == ["codex", "exec", "resume", "--last"] and resumed[-3:] == ["--model", "m", "-"]


@pytest.mark.parametrize("agent", ["gemini", "kimi", "unknown"])
def test_other_agents_cannot_be_launched(agent):
    with pytest.raises(ValueError, match="cannot launch"):
        L.build_agent_cmd(agent)


def test_local_command_activates_the_virtualenv(tmp_path):
    ws = tmp_path / "ws"
    cmd = L.build_local_cmd(ws, ws / ".venv", ["claude", "--print"], env={"EXTRA": "1"})
    assert cmd[:5] == ["env", "-u", "PYTHONPATH", "-u", "PYTHONHOME"]
    assert isinstance(cmd[5], L.RawShellArg) and cmd[5] == f'PATH={ws}/.venv/bin:"$PATH"'
    assert f"VIRTUAL_ENV={ws}/.venv" in cmd and "EXTRA=1" in cmd
    assert f"PIP_CACHE_DIR={ws}/.cache/pip" in cmd
    # TMPDIR stays the machine's: a path inside the workspace is often too
    # long for the UNIX sockets that multiprocessing and others create there.
    assert not any(str(arg).startswith("TMPDIR=") for arg in cmd)
    assert cmd[-2:] == ["claude", "--print"]


# ── Launcher script ──
# A stand-in for the agent: reads its prompt, optionally marks the run done.

ECHO_AGENT = ["bash", "-c", "cat"]
DONE_AGENT = ["bash", "-c", 'cat; echo \'{"done": true}\' > status.json']


def _script(tmp_path, cmd=None, **kwargs):
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    run_directory = tmp_path / ".sciworld" / "ws"
    kwargs.setdefault("main_prompt_text", "MAIN")
    script = L.write_launch_script(
        run_directory / "launch.sh", ws, cmd or DONE_AGENT, run_directory / "agent.log", **kwargs)
    return script, ws


def _run(script, **env):
    return subprocess.run(["bash", str(script)], capture_output=True, text=True,
                          env=dict(os.environ, SCIWORLD_RETRY_DELAY="0", **env), timeout=120)


def _source(tmp_path):
    source = tmp_path / "source"
    (source / "eval" / "test").mkdir(parents=True)
    (source / "eval" / "train").mkdir()
    (source / "eval" / "test" / "labels.json").write_text("{}")
    (source / "scripts").mkdir()
    (source / "scripts" / "evaluate_test.sh").write_text("#!/bin/bash\necho held-out\n")
    return source


def test_script_is_private_and_valid_bash(tmp_path):
    script, _ = _script(tmp_path, resume_cmd=["cat"], max_loops=4, test_phase_cmd=["cat"],
                        test_source_dir=tmp_path / "source", ro_flag="--bind", localize=True,
                        env_file=tmp_path / "launch.env", traj_dir=tmp_path / "trajectory",
                        continue_prompt_text="CONT", test_prompt_text="TEST", done_hint="score it")
    assert stat.S_IMODE(script.stat().st_mode) == 0o700
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
    text = script.read_text()
    assert "MAX_LOOPS=4" in text and "test-phase \"$WORKSPACE\" \"$SOURCE_DIR\" --localize" in text
    assert "ro_mounts READMEs eval baseline" in text
    # Prompts are files next to the script, not shell text.
    assert "MAIN" not in text and "TEST" not in text
    assert (script.parent / "prompt_main.md").read_text() == "MAIN\n"
    assert (script.parent / "prompt_continue.md").read_text() == "CONT\n"
    assert (script.parent / "prompt_test.md").read_text() == "TEST\n"
    # Nothing in the script needs bash 4 (macOS ships 3.2).
    for construct in ("mapfile", "readarray", "&>>", "${!", ",,}", "^^}", "declare -A", "coproc"):
        assert construct not in text


@pytest.mark.parametrize("prompt", [
    "line one\n$HOME `id` $(id) \"quotes\" 'single' \\ backslash\nEOF\n! * ? ~",
    "Don't use more than one GPU.",          # an unpaired apostrophe
    "1) submit with srun\n2) wait :)",       # unpaired parentheses
    "a \"quote that never closes",
    "$(rm -rf /tmp/never) `false` ${UNSET:?}",
])
def test_prompt_text_reaches_the_agent_verbatim(tmp_path, prompt):
    script, ws = _script(tmp_path, cmd=["bash", "-c", "cat > received.txt; echo '{\"done\": true}' > status.json"],
                         main_prompt_text=prompt)
    result = _run(script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (ws / "received.txt").read_text() == prompt + "\n"


def test_exit_status_tells_whether_the_run_finished(tmp_path):
    finished, _ = _script(tmp_path / "a", cmd=DONE_AGENT)
    assert _run(finished).returncode == 0
    # The agent ended cleanly but never marked the run done.
    unfinished, _ = _script(tmp_path / "b", cmd=ECHO_AGENT)
    result = _run(unfinished)
    assert result.returncode == 1 and "did not finish" in result.stdout
    # The agent failed: its status is the launcher's.
    failing, _ = _script(tmp_path / "c", cmd=["bash", "-c", "cat >/dev/null; exit 7"])
    assert _run(failing).returncode == 7


def test_the_launchers_own_messages_are_in_the_log(tmp_path):
    script, _ = _script(tmp_path, cmd=["bash", "-c", "cat >/dev/null; echo boom; exit 5"],
                        resume_cmd=ECHO_AGENT, max_loops=6, continue_prompt_text="go on")
    result = _run(script)
    assert result.returncode == 5
    log = (script.parent / "agent.log").read_text()
    assert log.count("boom") == 3
    for expected in ("[sciworld] started", "[sciworld] the agent exited with status 5",
                     "[sciworld] three failed invocations in a row; giving up.",
                     "[sciworld] the run did not finish."):
        assert expected in log, expected
    assert "\033[" not in "\n".join(l for l in log.splitlines() if l.startswith("[sciworld]"))


def test_launcher_records_its_pid_while_running_and_its_exit_status_after(tmp_path):
    script, ws = _script(tmp_path, cmd=["bash", "-c", "cat >/dev/null; sleep 30"])
    process = subprocess.Popen(["bash", str(script)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               start_new_session=True)
    try:
        deadline = time.time() + 30
        while not (script.parent / "launcher.pid").exists() and time.time() < deadline:
            time.sleep(0.05)
        state = rundir.launcher_state(ws)
        assert state["alive"] is True and state["pid"] == process.pid and state["pgid"] == process.pid
        assert state["exit"] is None
        # A second launcher on the same workspace refuses to start.
        second = _run(script)
        assert second.returncode == 75 and "already running this workspace" in second.stderr
        assert rundir.launcher_state(ws)["pid"] == process.pid  # and leaves the record alone
    finally:
        os.killpg(process.pid, 15)
        process.wait(timeout=30)
    state = rundir.launcher_state(ws)
    assert state["alive"] is False and state["pid"] is None
    assert state["exit"] == 143  # 128 + SIGTERM


def test_a_launcher_seen_as_gone_has_its_exit_status(tmp_path, monkeypatch):
    """The launcher removed its pid file before it wrote its exit status, and
    the state was read in the other order: for an instant, a run that had
    just ended looked gone without a status."""
    script, ws = _script(tmp_path)
    directory = script.parent
    # The status is written first, in one step; only then does the pid file go.
    trap = next(l for l in script.read_text().splitlines() if l.startswith("trap ") and l.endswith(" EXIT"))
    assert (trap.index('mv -f "$RUN_DIR/launcher.exit.tmp" "$RUN_DIR/launcher.exit"')
            < trap.index('rm -f "$RUN_DIR/launcher.pid"'))

    # A launcher that ends while its state is being read: the pid is looked
    # at first, so the status read after it is already there.
    (directory / "launcher.pid").write_text(
        f"pid={os.getpid()}\npgid={os.getpid()}\nhost={os.uname().nodename}\n")

    def ends_during_the_probe(pid, marker):
        (directory / "launcher.exit").write_text("0\n")
        (directory / "launcher.pid").unlink()
        return False

    with monkeypatch.context() as patched:
        patched.setattr(rundir, "_process_runs", ends_during_the_probe)
        state = rundir.launcher_state(ws)
    assert state["alive"] is False and state["exit"] == 0
    (directory / "launcher.exit").unlink()

    # And a real launcher, watched as closely as possible while it ends.
    process = subprocess.Popen(["bash", str(script)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    seen = False
    deadline = time.time() + 60
    try:
        while time.time() < deadline:
            state = rundir.launcher_state(ws)
            seen = seen or state["pid"] is not None
            if state["exit"] is not None:
                break
            assert not (seen and state["alive"] is False), state
    finally:
        process.wait(timeout=60)
    assert rundir.launcher_state(ws) == {"alive": False, "pid": None, "pgid": None, "host": None, "exit": 0}
    assert not (directory / "launcher.exit.tmp").exists()


def test_a_stale_pid_record_does_not_block_a_new_launcher(tmp_path):
    script, ws = _script(tmp_path)
    (script.parent / "launcher.pid").write_text(f"pid=999999\npgid=999999\nhost={os.uname().nodename}\n")
    assert rundir.launcher_state(ws)["alive"] is False
    assert _run(script).returncode == 0
    assert rundir.launcher_state(ws)["exit"] == 0
    # A record from another host cannot be checked from here: the launcher
    # refuses, and the state says so.
    (script.parent / "launcher.pid").write_text("pid=4242\npgid=4242\nhost=some-other-node\n")
    state = rundir.launcher_state(ws)
    assert state["alive"] is None and state["host"] == "some-other-node"
    refused = _run(script)
    assert refused.returncode == 75 and "some-other-node" in refused.stderr


def test_read_only_mounts_are_decided_before_every_invocation(tmp_path):
    """A stand-in runtime prints its arguments, so the mounts of each phase
    can be read from the output."""
    ws = tmp_path / "ws"
    (ws / "eval" / "train").mkdir(parents=True)
    (ws / "method").mkdir()
    (ws / "READMEs").mkdir()
    source = _source(tmp_path)
    # The stand-in marks the run done, so the test phase follows, and writes
    # test scores, which is what completes that phase.
    runtime = ["bash", "-c", 'cat >/dev/null; echo "ARGS $*"; echo \'{"done": true}\' > status.json; '
                             'mkdir -p scoring; echo "{}" > scoring/scores_test.json', "runtime",
               L.RawShellArg('"${SW_RO[@]}"')]
    run_directory = tmp_path / ".sciworld" / "ws"
    script = L.write_launch_script(
        run_directory / "launch.sh", ws, runtime, run_directory / "agent.log", test_phase_cmd=runtime,
        test_source_dir=source, ro_flag="--bind", test_prompt_text="TEST",
    )
    result = _run(script)
    assert result.returncode == 0, result.stdout + result.stderr
    main_args, test_args = [l for l in result.stdout.splitlines() if l.startswith("ARGS")]
    assert main_args == (
        f"ARGS --bind {ws}/READMEs:/home/user/READMEs:ro --bind {ws}/eval:/home/user/eval:ro"
    )
    # scores_train.json does not exist, so it is not mounted; the rest is.
    assert test_args == (
        f"ARGS --bind {ws}/READMEs:/home/user/READMEs:ro --bind {ws}/eval:/home/user/eval:ro"
        f" --bind {ws}/scripts/evaluate_test.sh:/home/user/scripts/evaluate_test.sh:ro"
        f" --bind {ws}/method:/home/user/method:ro"
    )
    assert (ws / "eval" / "test").is_dir()
    assert (ws / ".cache" / "tmp").is_dir()  # where the container's TMPDIR points


def test_a_planted_link_stops_the_run_instead_of_exposing_the_host(tmp_path):
    ws = tmp_path / "ws"
    (ws / "method").mkdir(parents=True)
    source = _source(tmp_path)
    host_dir = tmp_path / "host_home"
    host_dir.mkdir()
    (host_dir / "id_rsa").write_text("PRIVATE KEY")
    # Main phase: the "agent" swaps method/ for a link to a host directory.
    plant = ["bash", "-c", f'cat >/dev/null; echo "ARGS $*"; rm -rf method; ln -s {host_dir} method; '
                           'echo \'{"done": true}\' > status.json', "runtime", L.RawShellArg('"${SW_RO[@]}"')]
    run_directory = tmp_path / ".sciworld" / "ws"
    script = L.write_launch_script(
        run_directory / "launch.sh", ws, plant, run_directory / "agent.log", test_phase_cmd=plant,
        test_source_dir=source, ro_flag="--bind", test_prompt_text="TEST",
    )
    result = _run(script)
    assert result.returncode == 3, result.stdout + result.stderr
    assert len([l for l in result.stdout.splitlines() if l.startswith("ARGS")]) == 1  # no second start
    assert str(host_dir) not in result.stdout.replace(f"ln -s {host_dir}", "")
    log = (run_directory / "agent.log").read_text()
    assert "leads outside" in log and "method" in log
    assert rundir.launcher_state(ws)["exit"] == 3


def test_test_phase_never_writes_through_links_the_agent_left(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    source = _source(tmp_path)
    victim_dir = tmp_path / "host"
    victim_dir.mkdir()
    bashrc = victim_dir / "bashrc"
    bashrc.write_text("# my shell setup\n")
    notes = victim_dir / "notes.txt"
    notes.write_text("important\n")
    plant = ["bash", "-c",
             f'cat >/dev/null; mkdir -p scripts; ln -sf {bashrc} scripts/evaluate_test.sh; '
             f'ln -sf {notes} status.json; ln -sf {bashrc} agent.log; echo planted; '
             # status.json is now a link: writing "done" through it marks the run done
             'echo \'{"done": true}\' > status.json']
    finish = ["bash", "-c", 'cat >/dev/null; mkdir -p scoring; echo "{}" > scoring/scores_test.json']
    script, _ = _script(tmp_path, cmd=plant, test_phase_cmd=finish, test_source_dir=source,
                        test_prompt_text="TEST")
    result = _run(script)
    assert result.returncode == 0, result.stdout + result.stderr
    # The host files were not written by the launcher: bashrc is untouched and
    # notes.txt holds only what the agent itself wrote through its own link.
    assert bashrc.read_text() == "# my shell setup\n"
    assert notes.read_text() == '{"done": true}\n'
    installed = ws / "scripts" / "evaluate_test.sh"
    assert not installed.is_symlink() and "held-out" in installed.read_text()
    assert (ws / "status.json").is_symlink()   # the agent's own file: the launcher does not write it
    assert "planted" in (script.parent / "agent.log").read_text()


WRITE_TEST_SCORES = 'mkdir -p scoring; echo "{}" > scoring/scores_test.json'


def test_code_left_in_the_workspace_never_runs_as_the_launcher(tmp_path):
    """The launcher called its own Python modules from inside the workspace.
    Python looks for modules in the current directory first, so a `json.py`
    or a `sciworld/` directory left there by the agent ran on the host, as
    the user."""
    marker = tmp_path / "planted-code-ran"
    payload = f"open({str(marker)!r}, 'a').write(__name__ + '\\n')\n"
    plant = (
        "import pathlib, sys; sys.stdin.read(); pathlib.Path('sciworld').mkdir(exist_ok=True); "
        f"[pathlib.Path(p).write_text({payload!r}) for p in "
        "('json.py', 'sciworld/__init__.py', 'sciworld/guard.py', 'sciworld/format_stream.py')]"
    )
    # The first invocation plants the files and stops short. Every helper
    # the launcher calls from then on (done check, output formatter, test
    # phase) starts while they are there.
    script, ws = _script(
        tmp_path, cmd=[sys.executable, "-c", plant], resume_cmd=DONE_AGENT, max_loops=3,
        continue_prompt_text="go on", traj_dir=tmp_path / ".sciworld" / "ws" / "trajectory",
        test_source_dir=_source(tmp_path), test_prompt_text="TEST",
        test_phase_cmd=["bash", "-c", f"cat >/dev/null; {WRITE_TEST_SCORES}"])
    result = _run(script)
    assert (ws / "json.py").exists() and (ws / "sciworld" / "guard.py").exists()   # the files were planted
    assert not marker.exists(), "planted code ran: " + marker.read_text()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test phase complete" in result.stdout


def test_the_held_out_scores_complete_the_test_phase(tmp_path):
    """What a real agent did: it ran the held-out evaluator and wrote its
    reflection, but left status.json as it had set it in the main phase."""
    script, ws = _script(tmp_path, cmd=DONE_AGENT, test_source_dir=_source(tmp_path), test_prompt_text="TEST",
                         test_phase_cmd=["bash", "-c", f"cat >/dev/null; {WRITE_TEST_SCORES}"])
    result = _run(script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test phase complete" in result.stdout
    assert (ws / "status.json").read_text().strip() == '{"done": true}'   # never rewritten by the launcher
    assert rundir.launcher_state(ws)["exit"] == 0


@pytest.mark.parametrize("test_phase, status, message", [
    # The agent says it is done but the held-out evaluator never ran.
    ('cat >/dev/null; echo \'{"done": true}\' > status.json', 1, "ended without scoring/scores_test.json"),
    # The agent failed before the evaluator ran.
    ("cat >/dev/null; exit 7", 7, "ended without scoring/scores_test.json (exit status 7)"),
    # The scores are there, but the agent did not end cleanly.
    (f"cat >/dev/null; {WRITE_TEST_SCORES}; exit 9", 9, "the agent exited with status 9"),
])
def test_a_test_phase_without_its_result_is_not_a_finished_run(tmp_path, test_phase, status, message):
    script, ws = _script(tmp_path, cmd=DONE_AGENT, test_source_dir=_source(tmp_path), test_prompt_text="TEST",
                         test_phase_cmd=["bash", "-c", test_phase])
    result = _run(script)
    assert result.returncode == status, result.stdout + result.stderr
    assert message in result.stdout and "the run did not finish" in result.stdout
    assert rundir.launcher_state(ws)["exit"] == status


def test_test_scores_written_before_the_held_out_evaluator_do_not_count(tmp_path):
    made_up = ["bash", "-c", f'cat >/dev/null; {WRITE_TEST_SCORES}; echo \'{{"done": true}}\' > status.json']
    script, ws = _script(tmp_path, cmd=made_up, test_source_dir=_source(tmp_path), test_prompt_text="TEST",
                         test_phase_cmd=["bash", "-c", "cat >/dev/null"])
    result = _run(script)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "ended without scoring/scores_test.json" in result.stdout
    assert not (ws / "scoring" / "scores_test.json").exists()


def test_test_phase_does_not_run_without_the_held_out_evaluator(tmp_path):
    ws = tmp_path / "ws"
    (ws / "eval" / "train").mkdir(parents=True)
    (ws / "eval" / "train" / "evaluate.py").write_text("visible\n")
    script, _ = _script(tmp_path, cmd=DONE_AGENT, test_phase_cmd=["bash", "-c", "cat; echo TEST-RAN"],
                        test_source_dir=tmp_path / "source-that-was-deleted", test_prompt_text="TEST")
    result = _run(script)
    assert result.returncode == 4, result.stdout + result.stderr
    assert "TEST-RAN" not in result.stdout and "the test phase did not run" in result.stdout
    assert (ws / "eval" / "train" / "evaluate.py").read_text() == "visible\n"  # not deleted
    assert "no held-out evaluator" in (script.parent / "agent.log").read_text()


def test_env_file_is_private_and_quoted(tmp_path):
    path = L.write_env_file(tmp_path / "d" / "launch.env", {"KEY": "va'l ue$x", "OTHER": "1"})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    out = subprocess.run(["bash", "-c", f'. {path}; printf "%s|%s" "$KEY" "$OTHER"'],
                         capture_output=True, text=True)
    assert out.stdout == "va'l ue$x|1"


# ── Sessions ──

def test_unknown_session_manager(tmp_path):
    ok, message = L.launch_in_session("zellij", "n", tmp_path / "x", tmp_path / "log")
    assert not ok and "unknown session manager" in message


def test_missing_tmux_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(L.shutil, "which", lambda name: None)
    assert L.launch_in_session("tmux", "n", tmp_path / "x", tmp_path / "log") == (False, "tmux not found in PATH")
    assert L.default_session() == "nohup"
    assert L.session_exists("tmux", "n") is False and L.screen_sessions("n") == []


def _sleeper(tmp_path, name):
    script = tmp_path / f"{name}.sh"
    script.write_text("#!/bin/bash\nsleep 60\n")
    return script


def _wait_for(condition, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.1)
    return False


@pytest.mark.parametrize("manager", ["tmux", "screen"])
def test_starting_a_run_never_ends_another_runs_session(tmp_path, manager):
    """`tmux kill-session -t NAME` and `screen -S NAME` match by prefix. A run
    named like the beginning of another run's name must leave it alone."""
    if not shutil.which(manager):
        pytest.skip(f"{manager} is not installed")
    longer, shorter = "sw_2401_00001_claude-code-innovate-v2", "sw_2401_00001_claude-code-innovate"
    ok, message = L.launch_in_session(manager, longer, _sleeper(tmp_path, "a"), tmp_path / "a.log")
    assert ok, message
    assert _wait_for(lambda: L.session_exists(manager, longer))
    assert not L.session_exists(manager, shorter)  # a prefix is not a match

    ok, message = L.launch_in_session(manager, shorter, _sleeper(tmp_path, "b"), tmp_path / "b.log")
    assert ok, message
    assert _wait_for(lambda: L.session_exists(manager, shorter))
    assert L.session_exists(manager, longer), "the other run's session was ended"

    # A session of the same name is a run that is still going: never replaced.
    ok, message = L.launch_in_session(manager, shorter, _sleeper(tmp_path, "c"), tmp_path / "c.log")
    assert not ok and "already exists" in message
    assert L.session_exists(manager, shorter) and L.session_exists(manager, longer)

    # The printed stop command ends exactly one session.
    subprocess.run(L.stop_command(manager, shorter), shell=True, capture_output=True)
    assert _wait_for(lambda: not L.session_exists(manager, shorter))
    assert L.session_exists(manager, longer)


def test_wait_for_launcher_reports_what_happened(tmp_path):
    ok_script, ok_ws = _script(tmp_path / "ok", cmd=DONE_AGENT)
    assert L.launch_in_session("nohup", "n", ok_script, ok_script.parent / "agent.log")[0]
    assert L.wait_for_launcher(ok_ws, "nohup", "n") in (("running", None), ("finished", 0))
    assert _wait_for(lambda: rundir.launcher_state(ok_ws)["exit"] == 0)
    assert L.wait_for_launcher(ok_ws, "nohup", "n") == ("finished", 0)

    bad_script, bad_ws = _script(tmp_path / "bad", cmd=["bash", "-c", "cat >/dev/null; exit 9"])
    assert L.launch_in_session("nohup", "n", bad_script, bad_script.parent / "agent.log")[0]
    assert _wait_for(lambda: rundir.launcher_state(bad_ws)["exit"] is not None)
    assert L.wait_for_launcher(bad_ws, "nohup", "n") == ("failed", 9)

    slow_script, slow_ws = _script(tmp_path / "slow", cmd=["bash", "-c", "cat >/dev/null; sleep 30"])
    assert L.launch_in_session("nohup", "n", slow_script, slow_script.parent / "agent.log")[0]
    try:
        assert L.wait_for_launcher(slow_ws, "nohup", "n") == ("running", None)
    finally:
        os.killpg(rundir.launcher_state(slow_ws)["pgid"], 15)


# ── Credentials ──

def test_claude_and_codex_credentials_are_copied_privately(isolated, tmp_path):
    home = isolated / "home"
    (home / ".claude").mkdir()
    (home / ".claude" / ".credentials.json").write_text("{}")
    ws = tmp_path / "ws"
    notes = L.copy_agent_credentials("claude-code", ws)
    copied = ws / ".claude" / ".credentials.json"
    assert copied.exists() and stat.S_IMODE(copied.stat().st_mode) == 0o600
    assert not notes[0].startswith("WARNING")
    assert L.copy_agent_credentials("codex", ws)[0].startswith("WARNING")
    assert L.copy_agent_credentials("something-else", ws) == []


def test_credentials_are_never_written_through_a_link(isolated, tmp_path):
    """Between two turns the agent owns the workspace. A link at the place of
    the login file, or of its directory, must not redirect the copy."""
    home = isolated / "home"
    (home / ".claude").mkdir()
    (home / ".claude" / ".credentials.json").write_text('{"token": "secret"}')
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me\n")
    ws = tmp_path / "ws"
    (ws / ".claude").mkdir(parents=True)
    (ws / ".claude" / ".credentials.json").symlink_to(victim)
    L.copy_agent_credentials("claude-code", ws)
    assert victim.read_text() == "keep me\n"
    copied = ws / ".claude" / ".credentials.json"
    assert not copied.is_symlink() and "secret" in copied.read_text()

    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.rmtree(ws / ".claude")
    (ws / ".claude").symlink_to(outside)
    L.copy_agent_credentials("claude-code", ws)
    assert list(outside.iterdir()) == []
    assert (ws / ".claude").is_dir() and not (ws / ".claude").is_symlink()


def test_kimi_tokens_are_seeded_once(isolated, tmp_path):
    src = isolated / "home" / ".kimi"
    (src / "credentials").mkdir(parents=True)
    (src / "credentials" / "kimi-code.json").write_text("host-token")
    (src / "config.toml").write_text("a = 1")
    ws = tmp_path / "ws"
    L.copy_agent_credentials("kimi", ws)
    rotated = ws / ".kimi" / "credentials" / "kimi-code.json"
    rotated.write_text("rotated-in-workspace")
    (ws / ".kimi" / "kimi.json").write_text('{"work_dirs": ["/home/user"]}')
    L.copy_agent_credentials("kimi", ws)
    assert rotated.read_text() == "rotated-in-workspace"
    assert "/home/user" in (ws / ".kimi" / "kimi.json").read_text()


# ── Git and prompt ──

def test_git_repository_ignores_logins_and_caches(tmp_path):
    ws = tmp_path / "ws"
    (ws / ".claude").mkdir(parents=True)
    (ws / ".claude" / ".credentials.json").write_text("secret")
    (ws / "trajectory").mkdir()
    (ws / "trajectory" / "turn_0001_raw.jsonl").write_text("{}")
    (ws / "method").mkdir()
    (ws / "method" / "m.py").write_text("x = 1\n")
    assert L.init_git_repo(ws) is True
    tracked = subprocess.run(["git", "ls-files"], cwd=ws, capture_output=True, text=True).stdout.split()
    assert tracked == [".gitignore", "method/m.py"]
    assert L.init_git_repo(ws) is True  # a second call changes nothing
    log = subprocess.run(["git", "log", "--oneline"], cwd=ws, capture_output=True, text=True).stdout
    assert len(log.splitlines()) == 1


def test_prompt_composition():
    text = L.build_agent_prompt_text("SYS", "TASK", "run on one GPU", "## Environment\n\n- note")
    assert text == "SYS\n\nTASK\n\n## Compute\n\nrun on one GPU\n\n## Environment\n\n- note"
    assert L.build_agent_prompt_text("", "TASK", "", None) == "TASK"
