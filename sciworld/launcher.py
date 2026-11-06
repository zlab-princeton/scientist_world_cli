"""Build and launch the agent that works inside a workspace.

Four concerns, each in its own function:

``build_container_cmd``   isolation: container runtime, read-only binds, environment
``build_local_cmd``       venv mode: run on this machine in a per-workspace virtualenv
``build_agent_cmd``       the agent CLI invocation (the prompt arrives on stdin)
``write_launch_script``   the bash launcher: main phase, optional resume loop,
                          optional held-out test phase

``launch_in_session`` then starts the launcher under tmux, screen or nohup.

The launcher runs on the host with the user's permissions while the agent,
inside its container, can write anything into the workspace. Every host-side
step that reads or writes the workspace therefore goes through
:mod:`sciworld.guard`, and the launcher's own files (log, trajectory, prompts)
live in the run directory next to the workspace (:mod:`sciworld.rundir`).
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from sciworld.fsutil import ensure_real_dir, is_within, write_nofollow


# ── Read-only paths ───────────────────────────────────────────────────────

# Read-only for the whole run. Test-phase paths are not listed: they do not
# exist in the workspace during the main phase.
_WORKSPACE_RO_PATHS = [
    "READMEs",
    "eval/train",
    "baseline",
    "scripts/evaluate_train.sh",
    "scripts/baseline.sh",
]

# Additionally read-only during the test phase, once the held-out evaluator
# has been copied in. scoring/scores_train.json is the scientist's main-phase
# result and must not be rewritten while the test runs; the rest of scoring/
# stays writable so the evaluator can write scores_test.json.
_TEST_PHASE_RO_PATHS = [
    "eval/test",
    "scripts/evaluate_test.sh",
    "method",
    "scoring/scores_train.json",
]

# What the generated launcher protects. The list is evaluated before every
# agent invocation (see write_launch_script), so paths that appear later, such
# as the held-out evaluator, are covered too. The whole of eval/ is locked: in
# a workspace built by sciworld it holds only the evaluator.
LAUNCHER_MAIN_RO_PATHS = ["READMEs", "eval", "baseline",
                          "scripts/evaluate_train.sh", "scripts/baseline.sh"]
LAUNCHER_TEST_RO_PATHS = LAUNCHER_MAIN_RO_PATHS + [
    "scripts/evaluate_test.sh", "method", "scoring/scores_train.json",
]


# ── Blocked binaries ──────────────────────────────────────────────────────
# Apptainer shares the host's kernel and network, so a scheduler client that
# is visible in the container would drive the host's scheduler. These names
# are masked with /dev/null. A user who wants one of them can expose it with
# the `bind_paths` setting.
_BLOCKED_BINARIES = [
    # SLURM
    "sbatch", "srun", "salloc", "squeue", "scancel",
    "sinfo", "sacct", "scontrol", "sstat",
    # Nested container runtimes
    "apptainer", "singularity", "docker", "podman",
    # Privilege escalation
    "sudo", "su",
]

_BLOCK_TARGETS = ["/usr/bin", "/usr/local/bin", "/bin"]


_DEFAULT_ENV = {
    # CA bundle, so the agent can reach its model provider over HTTPS.
    "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
    "SSL_CERT_DIR": "/etc/ssl/certs",

    # Send heavy writes to the workspace (a bind mount with real disk behind
    # it) instead of the container's small in-memory writable layer.
    "TMPDIR": "/home/user/.cache/tmp",
    "PIP_CACHE_DIR": "/home/user/.cache/pip",
    "UV_CACHE_DIR": "/home/user/.cache/uv",

    # Make `pip install` land in the workspace, where it persists across
    # sessions. The two UV_ variables undo the image's system-wide defaults.
    "PIP_USER": "1",
    "PYTHONUSERBASE": "/home/user/.local",
    "UV_SYSTEM_PYTHON": "",
    "UV_BREAK_SYSTEM_PACKAGES": "",
}


_APPTAINER_FAMILY = {"apptainer", "singularity"}
_DOCKER_FAMILY = {"docker", "podman"}
SUPPORTED_RUNTIMES = ("apptainer", "singularity", "docker", "podman")

DOCKER_IMAGE_TAG = "sciworld-agent:latest"


class RawShellArg(str):
    """A command word that the launcher emits verbatim, without quoting, so a
    shell expansion such as ``"${SW_RO[@]}"`` or ``"$PATH"`` survives."""


def agent_image_ref(runtime: str, containers_dir: Path) -> str:
    """Where the agent image of *runtime* lives: a ``.sif`` path or a tag."""
    if runtime in _APPTAINER_FAMILY:
        return str(Path(containers_dir) / "agent.sif")
    if runtime in _DOCKER_FAMILY:
        return DOCKER_IMAGE_TAG
    raise ValueError(f"Unsupported container runtime: {runtime!r}")


def image_exists(runtime: str, image_ref: str) -> bool:
    """True if the agent image is present locally."""
    if runtime in _APPTAINER_FAMILY:
        return Path(image_ref).is_file()
    if runtime in _DOCKER_FAMILY:
        try:
            result = subprocess.run(
                [runtime, "image", "inspect", image_ref],
                capture_output=True, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0
    raise ValueError(f"Unsupported container runtime: {runtime!r}")


def selinux_enforcing(state_file: str = "/sys/fs/selinux/enforce") -> bool:
    """True on a host where SELinux is enforcing (the kernel's own flag)."""
    try:
        return Path(state_file).read_text().strip() == "1"
    except OSError:
        return False


def ro_bind_flag(runtime: str) -> str:
    """The flag that adds a bind mount for *runtime*."""
    if runtime in _APPTAINER_FAMILY:
        return "--bind"
    if runtime in _DOCKER_FAMILY:
        return "--volume"
    raise ValueError(f"Unsupported container runtime: {runtime!r}")


def container_env_prefix(runtime: str) -> Optional[str]:
    """Prefix that carries a host variable into a ``--cleanenv`` container.

    Apptainer and Singularity import ``<PREFIX>NAME`` as ``NAME``. Docker and
    Podman take ``--env NAME`` instead, so they have no prefix.
    """
    return {"apptainer": "APPTAINERENV_", "singularity": "SINGULARITYENV_"}.get(runtime)


# ── Container command ─────────────────────────────────────────────────────


def build_container_cmd(
    runtime: str,
    image: str,
    workspace: Path,
    extra_binds: Optional[list] = None,
    env: Optional[dict] = None,
    cmd: Optional[list] = None,
    *,
    test_phase: bool = False,
    test_phase_unlock_method: bool = False,
    empty_traj_overlay: Optional[Path] = None,
    extra_args: Optional[list] = None,
    protect: bool = True,
) -> list:
    """Return the argv that runs *cmd* inside the agent container.

    The contract is the same for all four runtimes: no host home directory,
    no host environment, an ephemeral ``/tmp``, the workspace mounted at
    ``/home/user``, and read-only mounts over the paths the scientist must
    not change.

    Args:
        runtime: ``apptainer``, ``singularity``, ``docker`` or ``podman``.
        image: path of the ``.sif`` (apptainer, singularity) or an image tag.
        workspace: host path of the workspace, mounted at ``/home/user``.
        extra_binds: ``(host_path, container_path)`` pairs, mounted read-only.
        env: extra environment variables for the container.
        cmd: the command to run inside the container.
        test_phase: also mount the test-phase paths that exist read-only.
        test_phase_unlock_method: with *test_phase*, leave ``method/`` writable.
        empty_traj_overlay: an empty host directory mounted read-only over
            ``/home/user/trajectory``, which hides the recorded trajectory
            from the agent.
        extra_args: runtime arguments inserted just before the image.
        protect: add the read-only mounts of the workspace paths that exist
            now. Pass ``False`` when the caller adds them itself.

    A workspace path that is a symbolic link to somewhere outside the
    workspace is never used as a mount source: the runtime would resolve it on
    the host and expose that location to the agent.
    """
    if runtime not in _APPTAINER_FAMILY and runtime not in _DOCKER_FAMILY:
        raise ValueError(
            f"Unsupported container runtime: {runtime!r}. "
            f"Supported: {', '.join(SUPPORTED_RUNTIMES)}."
        )
    workspace = Path(workspace)
    if any(c in str(workspace) for c in ":,"):
        raise ValueError(
            f"the workspace path {str(workspace)!r} contains ':' or ',', which "
            "container runtimes read as separators in mount specifications"
        )
    apptainer = runtime in _APPTAINER_FAMILY
    bind = ro_bind_flag(runtime)

    if apptainer:
        args = [runtime, "exec", "--containall", "--cleanenv"]
        # --no-eval (apptainer >= 1.1) stops the runtime from shell-evaluating
        # environment values. Singularity builds without it reject the flag.
        if runtime == "apptainer":
            args.append("--no-eval")
        args += [
            "--writable-tmpfs",
            "--home", f"{workspace}:/home/user",
            "--pwd", "/home/user",
        ]
    else:
        args = [runtime, "run", "--rm", "--interactive"]
        if selinux_enforcing():
            # SELinux hides bind-mounted host directories from a container
            # unless they are relabeled. Relabeling would change the user's
            # files, so label separation is switched off for this container.
            args += ["--security-opt", "label=disable"]
        if runtime == "podman" and os.geteuid() != 0:
            # Rootless podman maps the caller to root inside the container
            # unless told to keep the caller's own id.
            args.append("--userns=keep-id")
        args += [
            # Run as the calling user so files in the workspace stay owned by it.
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--tmpfs", "/tmp",
            "--workdir", "/home/user",
            "--env", "HOME=/home/user",
            "--volume", f"{workspace}:/home/user",
        ]

    # Host CA bundle, in case the image ships none.
    for candidate in ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt"):
        if os.path.exists(candidate):
            args += [bind, f"{candidate}:/etc/ssl/certs/ca-certificates.crt:ro"]
            break

    ro_paths: list = []
    if protect:
        ro_paths = list(_WORKSPACE_RO_PATHS)
        if test_phase:
            ro_paths += [
                p for p in _TEST_PHASE_RO_PATHS
                if not (test_phase_unlock_method and p == "method")
            ]
    for rel in ro_paths:
        host = workspace / rel
        if host.exists() and is_within(host, workspace):
            args += [bind, f"{host}:/home/user/{rel}:ro"]

    if empty_traj_overlay is not None and Path(empty_traj_overlay).exists():
        args += [bind, f"{empty_traj_overlay}:/home/user/trajectory:ro"]

    exposed = set()
    for host_path, container_path in (extra_binds or []):
        if Path(host_path).exists():
            args += [bind, f"{host_path}:{container_path}:ro"]
            exposed.add(str(container_path))

    # Docker and Podman containers only see the image's own binaries, so
    # nothing needs masking there. A path the caller mounted on purpose is
    # left alone.
    if apptainer:
        for name in _BLOCKED_BINARIES:
            for target_dir in _BLOCK_TARGETS:
                if f"{target_dir}/{name}" not in exposed:
                    args += ["--bind", f"/dev/null:{target_dir}/{name}:ro"]

    merged_env = dict(_DEFAULT_ENV)
    if env:
        merged_env.update({k: v for k, v in env.items() if v is not None})
    for key, value in merged_env.items():
        args += ["--env", f"{key}={value}"]

    if extra_args:
        args += list(extra_args)
    args.append(str(image))
    if cmd:
        args += list(cmd)
    return args


# ── Venv (no container) command ───────────────────────────────────────────


def build_local_cmd(
    workspace: Path,
    venv_dir: Path,
    cmd: Optional[list] = None,
    env: Optional[dict] = None,
) -> list:
    """Return the argv that runs *cmd* on this machine in venv mode.

    The workspace's virtualenv is activated through ``PATH`` and
    ``VIRTUAL_ENV``, so ``python``, ``pip`` and ``uv`` resolve to it while the
    agent CLI still comes from the host ``PATH``. ``HOME`` is untouched, so
    the agent uses the user's existing login. Package caches go under the
    workspace; ``TMPDIR`` is left as it is on this machine (a path inside the
    workspace is often too long for the sockets some libraries create there).

    This is **not a sandbox**: the agent runs with the user's permissions and
    can read and write anything the user can. Nothing in the workspace is
    write-protected. Use :func:`build_container_cmd` when isolation matters.
    """
    workspace = Path(workspace)
    venv_dir = Path(venv_dir)
    cache = workspace / ".cache"
    variables = {
        "PIP_CACHE_DIR": str(cache / "pip"),
        "UV_CACHE_DIR": str(cache / "uv"),
        "VIRTUAL_ENV": str(venv_dir),
        "UV_SYSTEM_PYTHON": "",
        "PYTHONUNBUFFERED": "1",
    }
    if env:
        variables.update({k: v for k, v in env.items() if v is not None})

    # A virtualenv is meant to be self-contained: an inherited PYTHONPATH
    # would shadow its site-packages.
    parts: list = ["env", "-u", "PYTHONPATH", "-u", "PYTHONHOME"]
    parts.append(RawShellArg(f'PATH={shlex.quote(str(venv_dir / "bin"))}:"$PATH"'))
    parts += [f"{key}={value}" for key, value in variables.items()]
    if cmd:
        parts += list(cmd)
    return parts


# ── Agent command ─────────────────────────────────────────────────────────

# Agents the launcher can drive from start to finish.
LAUNCHABLE_AGENTS = ("claude-code", "codex")


def build_agent_cmd(agent: str, *, resume: bool = False, model: Optional[str] = None) -> list:
    """Return the non-interactive agent invocation. The prompt is read from
    standard input, so its length and content never touch the command line.

    *resume* continues the most recent session of the working directory
    instead of starting a new one.
    """
    if agent == "claude-code":
        cmd = ["claude"]
        if resume:
            cmd.append("--continue")
        cmd += [
            "--print",
            "--dangerously-skip-permissions",
            "--verbose",
            "--output-format", "stream-json",
            # Load only explicitly configured MCP servers. Without this the
            # CLI also loads the connectors of the logged-in account.
            "--strict-mcp-config",
        ]
        if model:
            cmd += ["--model", model]
        return cmd
    if agent == "codex":
        cmd = ["codex", "exec"]
        if resume:
            cmd += ["resume", "--last"]
        cmd += [
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
        ]
        if model:
            cmd += ["--model", model]
        cmd.append("-")  # read the prompt from stdin
        return cmd
    raise ValueError(
        f"sciworld cannot launch agent {agent!r}; supported: {', '.join(LAUNCHABLE_AGENTS)}"
    )


# ── Launcher script ───────────────────────────────────────────────────────


def _emit_cmd_function(name: str, cmd: list) -> list:
    """Render *cmd* as a bash function, one option per line."""
    groups: list = []
    i = 0
    head: list = []
    while i < len(cmd) and not cmd[i].startswith("-"):
        head.append(cmd[i])
        i += 1
    if head:
        groups.append(head)
    while i < len(cmd):
        arg = cmd[i]
        if not arg.startswith("-"):
            groups.append(cmd[i:])  # image and inner command
            break
        takes_value = (
            "=" not in arg and i + 1 < len(cmd)
            and not cmd[i + 1].startswith("-")
            and not isinstance(cmd[i + 1], RawShellArg)
        )
        groups.append([arg, cmd[i + 1]] if takes_value else [arg])
        i += 2 if takes_value else 1

    def _render(arg: str) -> str:
        return arg if isinstance(arg, RawShellArg) else shlex.quote(arg)

    lines = [f"{name}() {{"]
    last = len(groups) - 1
    for j, group in enumerate(groups):
        indent = "    " if j == 0 else "        "
        suffix = " \\" if j < last else ""
        lines.append(f"{indent}{' '.join(_render(p) for p in group)}{suffix}")
    lines.append("}")
    return lines


def write_env_file(path: Path, variables: dict) -> Path:
    """Write ``export NAME='value'`` lines to *path*, readable by the owner only.

    The launcher sources this file, which keeps secrets such as API keys out
    of the launcher itself and out of process listings.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Environment for a sciworld run. Contains credentials: keep private."]
    for name, value in variables.items():
        lines.append(f"export {name}={shlex.quote(str(value))}")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    os.chmod(str(path), 0o600)
    return path


def write_launch_script(
    script_path: Path,
    workspace: Path,
    full_cmd: list,
    log_path: Path,
    *,
    resume_cmd: Optional[list] = None,
    max_loops: int = 1,
    status_file: str = "status.json",
    test_phase_cmd: Optional[list] = None,
    test_source_dir: Optional[Path] = None,
    main_prompt_text: str = "",
    continue_prompt_text: str = "",
    test_prompt_text: str = "",
    agent_name: str = "claude-code",
    traj_dir: Optional[Path] = None,
    python: Optional[str] = None,
    env_file: Optional[Path] = None,
    ro_flag: Optional[str] = None,
    localize: bool = False,
    done_hint: str = "",
) -> Path:
    """Write the bash launcher for one run and return its path.

    The launcher runs the main phase (``full_cmd`` with the main prompt),
    optionally resumes the session with ``resume_cmd`` until ``status.json``
    says ``done`` or *max_loops* is reached, and then, when *test_phase_cmd*
    and *test_source_dir* are given, installs the held-out evaluator in the
    workspace and runs the test phase.

    Files written next to the script, in the run directory: the prompts
    (``prompt_main.md``, ``prompt_continue.md``, ``prompt_test.md``), and at
    run time ``launcher.pid`` and ``launcher.exit``. Each command receives
    its prompt on standard input. Its output is piped through
    ``sciworld.format_stream``, which prints a readable transcript and records
    the raw trajectory under *traj_dir*; the transcript and the launcher's own
    messages are appended to *log_path*.

    The launcher exits with 0 when the run finished and with a non-zero
    status otherwise. Finished means that ``status.json`` reports ``done``
    after the main phase and, when there is a test phase, that the held-out
    evaluator wrote ``scoring/scores_test.json`` in it.

    Args:
        python: interpreter that has sciworld installed (default: this one).
        env_file: file of ``export`` lines sourced before anything runs.
        ro_flag: ``--bind`` or ``--volume``; set for container backends. The
            launcher then fills the bash array ``SW_RO`` with read-only mounts
            before every agent invocation, and the commands must reference it
            as ``"${SW_RO[@]}"``.
        localize: venv mode; rewrite ``/home/user`` in the held-out evaluator
            when it is installed.
        done_hint: a line printed when the run ends, e.g. how to score it.
    """
    script_path = Path(script_path)
    run_dir = script_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    python = python or sys.executable
    loops = max(1, int(max_loops)) if resume_cmd else 1
    test_enabled = bool(test_phase_cmd) and test_source_dir is not None

    # Prompts are files, read by the agent on standard input: their content
    # never passes through the shell.
    prompts = {"main": main_prompt_text}
    if resume_cmd:
        prompts["continue"] = continue_prompt_text
    if test_enabled:
        prompts["test"] = test_prompt_text
    for name, text in prompts.items():
        (run_dir / f"prompt_{name}.md").write_text(text.rstrip("\n") + "\n")

    fmt = ""
    if traj_dir is not None:
        fmt = (
            f' | sw_py -u -m sciworld.format_stream --agent {shlex.quote(agent_name)}'
            f' --traj-dir {shlex.quote(str(traj_dir))} --phase "$1"'
        )

    lines: list = [
        "#!/bin/bash",
        "# Launcher of one sciworld run, generated by `sw research`.",
        "# It runs in the foreground; `sw research` starts it under tmux,",
        "# screen or nohup. To run it by hand: bash <this file>",
        "#",
        "# Exit status: 0 when the run finished: status.json reports done after",
        "# the main phase and, if there is a test phase, the held-out evaluator",
        "# wrote scoring/scores_test.json. Otherwise the agent's exit status, or 1.",
        "",
        "set -o pipefail",
        "",
        f"PY={shlex.quote(str(python))}",
        f"WORKSPACE={shlex.quote(str(workspace))}",
        f"RUN_DIR={shlex.quote(str(run_dir))}",
        f"LOG={shlex.quote(str(log_path))}",
        f"STATUS={shlex.quote(status_file)}",
        f"MAX_LOOPS={loops}",
    ]
    if test_enabled:
        lines.append(f"SOURCE_DIR={shlex.quote(str(test_source_dir))}")
    if env_file is not None:
        lines += [
            "",
            "# Credentials and forwarded variables (owner-readable file).",
            f"ENV_FILE={shlex.quote(str(env_file))}",
            'if [ -f "$ENV_FILE" ]; then . "$ENV_FILE"; fi',
        ]

    lines += [
        "",
        "# ── Helpers ──",
        "",
        "# Messages go to the terminal in colour and to the log as plain text.",
        "sw_say()  { printf '\\033[%sm● sciworld\\033[0m  %s\\n' \"$1\" \"$2\"; "
        "printf '[sciworld] %s\\n' \"$2\" >> \"$LOG\"; }",
        "sw_info() { sw_say '1;96' \"$*\"; }",
        "sw_ok()   { sw_say '1;92' \"$*\"; }",
        "sw_warn() { sw_say '1;93' \"$*\"; }",
        "",
        "# Runs one of sciworld's own Python modules. Python puts the current",
        "# directory first on its module path, and the workspace belongs to the",
        "# agent: a json.py left there would be imported and run on this machine.",
        "# So these never run from the workspace, but from the run directory.",
        "sw_py() { ( cd \"$RUN_DIR\" && PYTHONSAFEPATH=1 exec \"$PY\" \"$@\" ); }",
        "",
        "# Prints yes when status.json holds {\"done\": true}.",
        "check_done() { sw_py -m sciworld.guard done \"$WORKSPACE\" \"$STATUS\"; }",
        "",
    ]
    if ro_flag:
        lines += [
            "# Fills SW_RO with read-only mounts for the listed workspace paths",
            "# that exist right now. A path that leads out of the workspace",
            "# through a symbolic link would expose the host: the run stops.",
            "ro_mounts() {",
            "    SW_RO=()",
            "    local listing rel",
            "    if ! listing=$(sw_py -m sciworld.guard mounts \"$WORKSPACE\" \"$@\" 2>> \"$LOG\"); then",
            "        sw_warn \"refusing to start the agent: a protected path in the workspace leads outside it. See $LOG\"",
            "        exit 3",
            "    fi",
            "    while IFS= read -r rel; do",
            "        if [ -n \"$rel\" ]; then",
            f"            SW_RO+=({ro_flag} \"$WORKSPACE/$rel:/home/user/$rel:ro\")",
            "        fi",
            "    done <<< \"$listing\"",
            "}",
            "",
        ]
    lines += [
        "# run_agent PHASE COMMAND PROMPT_FILE: runs one agent invocation, sets RC.",
        "run_agent() {",
        "    local fn=\"$2\"",
        f"    \"$fn\" < \"$3\" 2>&1{fmt} | tee -a \"$LOG\"",
        "    RC=${PIPESTATUS[0]}",
        "}",
        "",
        "# ── Agent commands ──",
        "",
    ]
    lines += _emit_cmd_function("cmd_main", full_cmd)
    if resume_cmd:
        lines.append("")
        lines += _emit_cmd_function("cmd_resume", resume_cmd)
    if test_enabled:
        lines.append("")
        lines += _emit_cmd_function("cmd_test", test_phase_cmd)

    main_ro = " ".join(shlex.quote(p) for p in LAUNCHER_MAIN_RO_PATHS)
    test_ro = " ".join(shlex.quote(p) for p in LAUNCHER_TEST_RO_PATHS)
    lines += [
        "",
        "# ── Start ──",
        "",
        'cd "$WORKSPACE" || { echo "sciworld: workspace not found: $WORKSPACE" >&2; exit 1; }',
        'mkdir -p "$RUN_DIR" "$(dirname "$LOG")"',
        "",
        "# One launcher per workspace. launcher.pid exists while one runs.",
        'HOST=$(hostname)',
        'if [ -f "$RUN_DIR/launcher.pid" ]; then',
        "    other_pid=$(sed -n 's/^pid=//p' \"$RUN_DIR/launcher.pid\")",
        "    other_host=$(sed -n 's/^host=//p' \"$RUN_DIR/launcher.pid\")",
        '    if [ -n "$other_host" ] && [ "$other_host" != "$HOST" ]; then',
        '        echo "sciworld: a launcher started on $other_host may still be running this workspace." >&2',
        '        echo "sciworld: if it is not, delete $RUN_DIR/launcher.pid and start again." >&2',
        "        exit 75",
        '    elif [ -n "$other_pid" ] && kill -0 "$other_pid" 2>/dev/null; then',
        '        echo "sciworld: another launcher (pid $other_pid) is already running this workspace." >&2',
        "        exit 75",
        "    fi",
        "fi",
        "printf 'pid=%s\\npgid=%s\\nhost=%s\\n' \"$$\" \"$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')\" \"$HOST\" "
        "> \"$RUN_DIR/launcher.pid\"",
        'rm -f "$RUN_DIR/launcher.exit"',
        "# The exit status is recorded before launcher.pid goes away, and in one",
        "# step: whoever sees this launcher as gone finds its status.",
        "trap 'code=$?; printf \"%s\\n\" \"$code\" > \"$RUN_DIR/launcher.exit.tmp\" "
        "&& mv -f \"$RUN_DIR/launcher.exit.tmp\" \"$RUN_DIR/launcher.exit\"; "
        "rm -f \"$RUN_DIR/launcher.pid\"' EXIT",
        "# Without these, bash would run the EXIT trap with the status of the",
        "# last finished command (0) when the launcher is stopped by a signal.",
        "trap 'exit 129' HUP",
        "trap 'exit 130' INT",
        "trap 'exit 143' TERM",
        "",
        'sw_info "started   $(date \'+%Y-%m-%d %H:%M:%S %Z\') on $HOST"',
        'sw_info "workspace $WORKSPACE"',
        'sw_info "log       $LOG"',
    ]
    if ro_flag:
        lines.append('sw_py -m sciworld.guard prepare "$WORKSPACE" 2>> "$LOG"')
    lines += [
        "",
        "# ── Main phase ──",
        "",
        "RC=0",
        "DONE=no",
        "STARTED=no     # becomes yes once an invocation exits cleanly",
        "FAILURES=0     # consecutive failed invocations",
        "i=1",
        'while [ "$i" -le "$MAX_LOOPS" ]; do',
    ]
    if ro_flag:
        lines.append(f"    ro_mounts {main_ro}")
    if resume_cmd:
        lines += [
            '    if [ "$STARTED" = yes ]; then',
            '        sw_info "iteration $i/$MAX_LOOPS: resuming the session"',
            '        run_agent main cmd_resume "$RUN_DIR/prompt_continue.md"',
            "    else",
            '        sw_info "iteration $i/$MAX_LOOPS: starting the agent"',
            '        run_agent main cmd_main "$RUN_DIR/prompt_main.md"',
            "    fi",
        ]
    else:
        lines += [
            '    sw_info "starting the agent"',
            '    run_agent main cmd_main "$RUN_DIR/prompt_main.md"',
        ]
    lines += [
        '    if [ "$RC" -eq 0 ]; then',
        "        STARTED=yes",
        "        FAILURES=0",
        "    else",
        "        FAILURES=$((FAILURES + 1))",
        '        sw_warn "the agent exited with status $RC"',
        "    fi",
        "    DONE=$(check_done)",
        '    if [ "$DONE" = yes ]; then',
        '        sw_ok "status.json reports done."',
        "        break",
        "    fi",
        '    if [ "$FAILURES" -ge 3 ]; then',
        '        sw_warn "three failed invocations in a row; giving up."',
        "        break",
        "    fi",
        '    if [ "$i" -ge "$MAX_LOOPS" ]; then',
        '        sw_warn "the agent stopped without setting done=true in $STATUS."',
        "        break",
        "    fi",
        '    [ "$RC" -eq 0 ] || sleep "${SCIWORLD_RETRY_DELAY:-10}"',
        "    i=$((i + 1))",
        "done",
    ]

    if test_enabled:
        install = 'sw_py -m sciworld.guard test-phase "$WORKSPACE" "$SOURCE_DIR"'
        if localize:
            install += " --localize"
        lines += [
            "",
            "# ── Test phase ──",
            "# Runs only after the main phase reported done. The held-out",
            "# evaluator is installed from the pristine source at this point; it",
            "# was never in the workspace before. The phase has done its job when",
            "# that evaluator has written scoring/scores_test.json.",
            "",
            'if [ "$DONE" = yes ]; then',
            '    sw_info "main phase complete: adding the held-out evaluator."',
            f'    if {install} >> "$LOG" 2>&1; then',
        ]
        if ro_flag:
            lines.append(f"        ro_mounts {test_ro}")
        lines += [
            '        sw_info "test phase: starting"',
            '        run_agent test cmd_test "$RUN_DIR/prompt_test.md"',
            '        SCORED=$(sw_py -m sciworld.guard test-scores "$WORKSPACE")',
            '        if [ "$SCORED" = yes ] && [ "$RC" -eq 0 ]; then',
            '            sw_ok "test phase complete."',
            '        elif [ "$SCORED" = yes ]; then',
            '            sw_warn "the held-out scores were written, but the agent exited with status $RC."',
            "            DONE=no",
            "        else",
            '            sw_warn "the test phase ended without scoring/scores_test.json (exit status $RC)."',
            "            DONE=no",
            "        fi",
            "    else",
            '        sw_warn "the held-out evaluator could not be installed from $SOURCE_DIR: the test phase did not run."',
            "        DONE=no",
            "        RC=4",
            "    fi",
            "else",
            '    sw_warn "the main phase did not finish: skipping the test phase."',
            "fi",
        ]

    lines += [
        "",
        'sw_info "finished  $(date \'+%Y-%m-%d %H:%M:%S %Z\')"',
        'if [ "$DONE" = yes ]; then',
    ]
    if done_hint:
        lines.append(f"    sw_info {shlex.quote(done_hint)}")
    lines += [
        "    exit 0",
        "fi",
        'sw_warn "the run did not finish."',
        '[ "$RC" -ne 0 ] && exit "$RC"',
        "exit 1",
        "",
    ]

    script_path.write_text("\n".join(lines))
    os.chmod(str(script_path), 0o700)
    return script_path


# ── Session (tmux / screen / nohup) ───────────────────────────────────────

SUPPORTED_SESSIONS = ("tmux", "screen", "nohup")


def default_session() -> str:
    """The best session manager available on this machine."""
    for name in ("tmux", "screen"):
        if shutil.which(name):
            return name
    return "nohup"


def tmux_session_exists(name: str) -> bool:
    """True if a tmux session is named exactly *name*.

    The ``=`` prefix matters: without it tmux also accepts a session whose
    name merely starts with *name*.
    """
    if not shutil.which("tmux"):
        return False
    try:
        return subprocess.run(
            ["tmux", "has-session", "-t", f"={name}"], capture_output=True, timeout=30,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def screen_sessions(name: str) -> list:
    """Ids (``<pid>.<name>``) of the screen sessions named exactly *name*.

    ``screen -S <name>`` matches by prefix; the full id does not.
    """
    if not shutil.which("screen"):
        return []
    try:
        listing = subprocess.run(["screen", "-ls"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    found = []
    for line in listing.splitlines():
        fields = line.split()
        if not fields:
            continue
        pid, sep, session = fields[0].partition(".")
        if sep and pid.isdigit() and session == name:
            found.append(fields[0])
    return found


def session_exists(session: str, name: str) -> bool:
    """True if *session* (``tmux`` or ``screen``) holds a session named exactly *name*."""
    if session == "tmux":
        return tmux_session_exists(name)
    if session == "screen":
        return bool(screen_sessions(name))
    return False


def _tmux_target(name: str) -> str:
    """``'=name'``: the exact-match form of a tmux target, always quoted
    because zsh expands an unquoted word that starts with ``=``."""
    return "'=" + name.replace("'", "") + "'"


def stop_command(session: str, name: str, pgid: Optional[int] = None) -> str:
    """The shell command that stops a run, for display."""
    if session == "tmux":
        return f"tmux kill-session -t {_tmux_target(name)}"
    if session == "screen":
        ids = screen_sessions(name)
        return f"screen -S {shlex.quote(ids[0] if ids else name)} -X quit"
    if pgid:
        return f"kill -- -{pgid}"
    return "kill the process group shown when the run was started"


def launch_in_session(
    session: str,
    session_name: str,
    launch_script: Path,
    log_path: Path,
) -> tuple:
    """Start ``bash <launch_script>`` detached, under *session*.

    An existing session of the same name is never touched: the launch fails
    instead, because that session belongs to a run that is still going.
    Returns ``(ok, message)``; the message tells the user how to follow or
    stop the run.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    wrapped = f"bash {shlex.quote(str(launch_script))}"

    if session == "tmux":
        if not shutil.which("tmux"):
            return False, "tmux not found in PATH"
        if tmux_session_exists(session_name):
            return False, (
                f"a tmux session named '{session_name}' already exists; it belongs to a run "
                f"that is still going. Stop it with: {stop_command('tmux', session_name)}"
            )
        result = subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name, "bash", "-lc", wrapped],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return False, f"tmux failed: {result.stderr.strip()}"
        return True, (
            f"Started in tmux session '{session_name}'.\n"
            f"  Attach:  tmux attach -t {_tmux_target(session_name)}   (detach with Ctrl-b d)\n"
            f"  Log:     tail -f {log_path}\n"
            f"  Stop:    {stop_command('tmux', session_name)}"
        )

    if session == "screen":
        if not shutil.which("screen"):
            return False, "screen not found in PATH"
        if screen_sessions(session_name):
            return False, (
                f"a screen session named '{session_name}' already exists; it belongs to a run "
                f"that is still going. Stop it with: {stop_command('screen', session_name)}"
            )
        result = subprocess.run(
            ["screen", "-dmS", session_name, "bash", "-lc", wrapped],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return False, f"screen failed: {result.stderr.strip()}"
        ids = screen_sessions(session_name)
        target = ids[0] if ids else session_name
        return True, (
            f"Started in screen session '{session_name}'.\n"
            f"  Attach:  screen -r {shlex.quote(target)}   (detach with Ctrl-a d)\n"
            f"  Log:     tail -f {log_path}\n"
            f"  Stop:    screen -S {shlex.quote(target)} -X quit"
        )

    if session == "nohup":
        try:
            process = subprocess.Popen(
                ["bash", "-lc", wrapped],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True,
            )
        except OSError as exc:
            return False, f"could not start the launcher: {exc}"
        _BACKGROUND[str(launch_script)] = process
        return True, (
            f"Started in the background (process group {process.pid}).\n"
            f"  Log:     tail -f {log_path}\n"
            f"  Stop:    kill -- -{process.pid}"
        )

    return False, f"unknown session manager {session!r}; expected one of {', '.join(SUPPORTED_SESSIONS)}"


# How long `sw research` waits for a launcher to report in, and how long it
# then watches it: a launcher that cannot work (missing agent, rejected
# credentials, runtime error) ends within moments.
LAUNCH_WAIT_SECONDS = 20.0
LAUNCH_SETTLE_SECONDS = 5.0

# Launchers started with nohup by this process, by script path.
_BACKGROUND: dict = {}


def wait_for_launcher(
    workspace: Path, session: str, session_name: str,
    timeout: Optional[float] = None, settle: Optional[float] = None,
) -> tuple:
    """Wait until the launcher of *workspace* has started, and report on it.

    Returns ``(state, exit_status)`` where *state* is ``"running"``,
    ``"finished"`` (it already ended with status 0), ``"failed"`` (it already
    ended with a non-zero status, or never started), or ``"starting"`` (it
    has not reported in within *timeout*, which happens when a login shell is
    slow to start).
    """
    from sciworld.rundir import launcher_state, run_dir

    timeout = LAUNCH_WAIT_SECONDS if timeout is None else timeout
    settle = LAUNCH_SETTLE_SECONDS if settle is None else settle
    process = _BACKGROUND.get(str(run_dir(workspace) / "launch.sh"))

    def holder_gone() -> bool:
        """True if what was started to hold the launcher has ended."""
        if session in ("tmux", "screen"):
            return not session_exists(session, session_name)
        return process is not None and process.poll() is not None

    # 1. Until the launcher reports in, or is known to be gone.
    seen = False
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = launcher_state(workspace)
        if state["pid"] is not None:
            seen = True
            break
        if state["exit"] is not None:
            break
        if holder_gone():
            time.sleep(0.3)  # give the exit record a moment to appear
            break
        time.sleep(0.1)
    # 2. Watch a launcher that came up for a moment longer.
    if seen:
        deadline = time.time() + settle
        while time.time() < deadline:
            state = launcher_state(workspace)
            if state["pid"] is None or state["alive"] is False:
                break
            time.sleep(0.1)

    state = launcher_state(workspace)
    if state["pid"] is not None and state["alive"] is not False:
        return "running", None
    if state["exit"] is not None:
        return ("finished" if state["exit"] == 0 else "failed"), state["exit"]
    if holder_gone():
        return "failed", (process.returncode if process is not None else None)
    return "starting", None


# ── Credentials ───────────────────────────────────────────────────────────


def _copy_private(src: Path, dst: Path, mode: int = 0o600, root: Optional[Path] = None) -> None:
    """Copy a login file into the workspace at *root*.

    The agent owns the workspace, so neither the directories on the way nor
    the destination itself may be a symbolic link that would send the file,
    and the user's credentials, somewhere else.
    """
    dst = Path(dst)
    if root is not None:
        ensure_real_dir(Path(root), str(dst.parent.relative_to(root)))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
    write_nofollow(dst, Path(src).read_bytes(), mode)


def copy_agent_credentials(agent: str, workspace: Path) -> list:
    """Copy the agent's login files from the host into *workspace*.

    Inside the container the workspace is ``$HOME``, and each agent CLI looks
    for its login there. Returns notes for the user; a note that starts with
    ``WARNING`` means no login was found on the host.
    """
    notes: list = []
    home = Path(os.path.expanduser("~"))
    workspace = Path(workspace)

    if agent == "claude-code":
        src = home / ".claude" / ".credentials.json"
        if src.exists():
            _copy_private(src, workspace / ".claude" / ".credentials.json", root=workspace)
            notes.append("Copied claude credentials to .claude/.credentials.json")
        else:
            notes.append(
                f"WARNING: {src} not found. Run `claude` on the host once to log in, "
                "or export ANTHROPIC_API_KEY."
            )

    elif agent == "codex":
        src = home / ".codex" / "auth.json"
        if src.exists():
            _copy_private(src, workspace / ".codex" / "auth.json", root=workspace)
            notes.append("Copied codex credentials to .codex/auth.json")
        else:
            notes.append(
                f"WARNING: {src} not found. Run `codex login` on the host once, "
                "or export CODEX_API_KEY."
            )

    elif agent == "kimi":
        src_dir = home / ".kimi"
        dst_dir = workspace / ".kimi"
        if not src_dir.exists():
            notes.append(
                f"WARNING: {src_dir} not found. Run `kimi login` on the host once to authenticate."
            )
        else:
            copied: list = []
            # Static state: safe to refresh on every call.
            for fname in ("config.toml", "device_id", "latest_version.txt"):
                src = src_dir / fname
                if src.exists():
                    _copy_private(src, dst_dir / fname, 0o600 if fname == "device_id" else 0o644, root=workspace)
                    copied.append(fname)
            # kimi.json is the session index that the CLI rewrites after every
            # turn. Only seed an empty one; copying the host's file again
            # would erase the session a later `--continue` needs.
            meta = dst_dir / "kimi.json"
            if not meta.exists():
                ensure_real_dir(workspace, ".kimi")
                write_nofollow(meta, b'{"work_dirs": []}', 0o644)
                copied.append("kimi.json (seed)")
            # OAuth tokens: copied once. The CLI rotates a single-use refresh
            # token inside the workspace; copying the host's file again would
            # overwrite the rotated token with a consumed one.
            src_creds = src_dir / "credentials"
            if src_creds.is_dir():
                for child in src_creds.iterdir():
                    target = dst_dir / "credentials" / child.name
                    if child.is_file() and not target.exists():
                        _copy_private(child, target, root=workspace)
                        copied.append(f"credentials/{child.name}")
            if copied:
                notes.append(f"Copied kimi credentials to .kimi/ ({', '.join(copied)})")
            else:
                notes.append(
                    f"WARNING: no auth files in {src_dir}. Run `kimi login` on the host once to authenticate."
                )

    elif agent == "gemini":
        src_dir = home / ".gemini"
        dst_dir = workspace / ".gemini"
        copied = []
        for fname in ("oauth_creds.json", "google_accounts.json", "projects.json",
                      "settings.json", "installation_id"):
            src = src_dir / fname
            if src.exists():
                _copy_private(src, dst_dir / fname, 0o600 if fname == "oauth_creds.json" else 0o644, root=workspace)
                copied.append(fname)
        if copied:
            notes.append(f"Copied gemini credentials to .gemini/ ({', '.join(copied)})")
        else:
            notes.append(
                f"WARNING: no auth files in {src_dir}. Run `gemini` on the host once to authenticate."
            )

    return notes


# ── Git ───────────────────────────────────────────────────────────────────

_GITIGNORE = """\
# Agent logins and tool state: never commit these.
.claude/
.claude.json
.codex/
.gemini/
.kimi/

# Caches, installed packages, virtualenv
.cache/
.local/
.config/
.npm/
.venv/
__pycache__/
*.pyc

# Run records kept outside version control
trajectory/
agent.log
jobs/
tmp/
*.log
.DS_Store
"""


def init_git_repo(workspace: Path) -> bool:
    """Make *workspace* a git repository with one initial commit.

    Returns ``True`` on success, and for a workspace that already is a
    repository. The identity is set locally, so no global git configuration
    is needed.
    """
    workspace = Path(workspace)
    if not shutil.which("git"):
        return False
    if (workspace / ".git").exists():
        return True

    def _git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=str(workspace), capture_output=True, text=True)

    if _git("init", "-q", "-b", "main").returncode != 0:
        _git("init", "-q")  # git older than 2.28 has no -b
    if not (workspace / ".git").exists():
        return False
    _git("config", "user.email", "agent@sciworld.local")
    _git("config", "user.name", "Sciworld Agent")
    gitignore = workspace / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(_GITIGNORE)
    _git("add", "-A")
    return _git("commit", "-q", "-m", "sciworld: initial environment").returncode == 0


# ── Prompt ────────────────────────────────────────────────────────────────


def build_agent_prompt_text(
    system_prompt: str,
    mode_prompt: str,
    compute_instructions: Optional[str] = None,
    environment_note: Optional[str] = None,
) -> str:
    """Compose the prompt the agent receives when a run starts.

    In order: the user's system prompt, the task prompt of the mode, the
    user's compute instructions, and a short note about the execution
    environment. Empty parts are left out.
    """
    parts: list = []
    if system_prompt and system_prompt.strip():
        parts.append(system_prompt.strip())
    parts.append(mode_prompt.strip())
    if compute_instructions and compute_instructions.strip():
        parts.append("## Compute\n\n" + compute_instructions.strip())
    if environment_note and environment_note.strip():
        parts.append(environment_note.strip())
    return "\n\n".join(parts)
