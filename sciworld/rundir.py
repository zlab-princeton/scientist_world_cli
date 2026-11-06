"""What sciworld records about a run, kept outside the workspace.

A run has two directories side by side::

    <parent>/<run>/               the workspace: all the agent can see and change
    <parent>/.sciworld/<run>/     the run directory: sciworld's own files

The run directory holds::

    run.json            paper, mode, agent, backend, session (written by sw research)
    launch.sh           the launcher
    launch.env          credentials and forwarded variables (owner-readable)
    prompt_*.md         the prompts the launcher sends
    agent.log           transcript of the run
    trajectory/         raw event stream of every agent invocation
    launcher.pid        written by a running launcher, removed when it exits
    launcher.exit       exit status of the last launcher, written before
                        launcher.pid is removed

Nothing in it is mounted into the agent's container, so the agent cannot read
its own transcript or redirect what the launcher writes.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Optional

RUN_DIR_NAME = ".sciworld"


def run_dir(workspace: Path) -> Path:
    """The run directory of the workspace at *workspace*."""
    workspace = Path(workspace)
    return workspace.parent / RUN_DIR_NAME / workspace.name


def ensure_run_dir(workspace: Path) -> Path:
    """Create the run directory (owner-only) and return it."""
    path = run_dir(workspace)
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(str(path), 0o700)
    return path


def log_path(workspace: Path) -> Path:
    """Where the transcript of the run is, or would be, written."""
    return run_dir(workspace) / "agent.log"


def existing_log(workspace: Path) -> Optional[Path]:
    """The transcript of the run if there is one.

    Runs started by sciworld 0.2 kept it inside the workspace.
    """
    for candidate in (log_path(workspace), Path(workspace) / "agent.log"):
        if candidate.is_file():
            return candidate
    return None


# ── run.json ──────────────────────────────────────────────────────────────


def write_record(workspace: Path, record: dict) -> Path:
    """Write ``run.json``; values that are not JSON types are stringified."""
    path = ensure_run_dir(workspace) / "run.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, default=str) + "\n")
    os.replace(tmp, path)
    return path


def read_record(workspace: Path) -> dict:
    """The content of ``run.json``, or ``{}`` when the run has none."""
    try:
        data = json.loads((run_dir(workspace) / "run.json").read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# ── Launcher state ────────────────────────────────────────────────────────


def _read_pid_file(path: Path) -> dict:
    fields: dict = {}
    try:
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep:
                fields[key.strip()] = value.strip()
    except (OSError, ValueError):
        return {}
    return fields


def _process_runs(pid: int, marker: str) -> bool:
    """True if process *pid* exists and its command line contains *marker*."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # exists, owned by someone else
    except OSError:
        return False
    cmdline = ""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "args="], capture_output=True, text=True, timeout=10,
            )
            cmdline = result.stdout
        except (OSError, subprocess.TimeoutExpired):
            return True  # cannot inspect it; the pid exists
    if not cmdline.strip():
        return True
    return marker in cmdline


def launcher_state(workspace: Path) -> dict:
    """What is known about the launcher of a run.

    Returns ``{"alive", "pid", "pgid", "host", "exit"}``. ``alive`` is
    ``True`` or ``False`` when it can be decided on this machine, and ``None``
    when the launcher was started on another host, or when the run has no
    records at all (it was not started by this version of sciworld).
    ``exit`` is the exit status of the last launcher that ended, if recorded.
    """
    directory = run_dir(workspace)
    state = {"alive": None, "pid": None, "pgid": None, "host": None, "exit": None}
    _probe_launcher(directory, state)
    # Read last. A launcher records its exit status before it removes its pid
    # file, so one that the probe found gone has its status on disk by now.
    try:
        state["exit"] = int((directory / "launcher.exit").read_text().strip())
    except (OSError, ValueError):
        pass
    return state


def _probe_launcher(directory: Path, state: dict) -> None:
    """Fill ``alive``, ``pid``, ``pgid`` and ``host`` of *state* from
    ``launcher.pid`` in *directory*."""
    pid_file = directory / "launcher.pid"
    if not pid_file.is_file():
        if directory.is_dir():
            state["alive"] = False
        return
    fields = _read_pid_file(pid_file)
    try:
        state["pid"] = int(fields.get("pid", ""))
    except ValueError:
        state["alive"] = False
        return
    try:
        state["pgid"] = int(fields.get("pgid", ""))
    except ValueError:
        pass
    state["host"] = fields.get("host") or None
    if state["host"] and state["host"] != socket.gethostname():
        return  # running elsewhere, as far as the record says
    # The launcher runs as `bash <path>/launch.sh`; the path may be relative
    # when it was started by hand, so only the file name is matched.
    state["alive"] = _process_runs(state["pid"], "launch.sh")
