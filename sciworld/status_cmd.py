"""``sw status``: show the runs on this machine and whether they are done."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Optional

from sciworld import rundir
from sciworld import style as S
from sciworld.settings import ConfigError, load_config
from sciworld.evaluate_cmd import list_runs
from sciworld.launcher import screen_sessions, tmux_session_exists
from sciworld.papers import normalize_arxiv_id
from sciworld.session import session_name


def _legacy_session_alive(name: str) -> Optional[bool]:
    """For runs without launcher records (started by sciworld 0.2): whether a
    tmux or screen session of exactly that name exists; ``None`` when neither
    tool is installed, so the run cannot be probed."""
    if not shutil.which("tmux") and not shutil.which("screen"):
        return None
    return tmux_session_exists(name) or bool(screen_sessions(name))


def run_status(arxiv_id: str, run_dir: Path) -> dict:
    """Summarize one run directory.

    ``state`` is one of ``running``, ``test phase``, ``finished``,
    ``finished (no test scores)``, ``failed``, ``stopped``, ``not started``.
    """
    done = False
    try:
        done = json.loads((run_dir / "status.json").read_text()).get("done") is True
    except (OSError, ValueError, AttributeError):
        pass
    scoring = run_dir / "scoring"
    has_train = (scoring / "scores_train.json").is_file()
    has_test = (scoring / "scores_test.json").is_file()
    in_test_phase = (run_dir / "scripts" / "evaluate_test.sh").exists()
    log = rundir.existing_log(run_dir)
    last_activity = None
    if log is not None:
        try:
            last_activity = log.stat().st_mtime
        except OSError:
            pass

    launcher = rundir.launcher_state(run_dir)
    alive = launcher["alive"]
    elsewhere = launcher["pid"] is not None and alive is None
    if alive is None and launcher["pid"] is None:
        # No launcher records: a run started by an older version, or by hand.
        alive = _legacy_session_alive(session_name(arxiv_id, run_dir.name))
        if alive is None and last_activity is not None and not done:
            alive = time.time() - last_activity < 600

    if alive or elsewhere:
        state = "test phase" if in_test_phase else "running"
    elif done and (has_test or not in_test_phase):
        state = "finished"
    elif done:
        state = "finished (no test scores)"
    elif last_activity is None and launcher["exit"] is None:
        state = "not started"
    elif launcher["exit"] not in (None, 0):
        state = "failed"
    else:
        state = "stopped"
    return {
        "arxiv_id": arxiv_id, "run": run_dir.name, "state": state, "done": done,
        "scores_train": has_train, "scores_test": has_test,
        "last_activity": last_activity, "path": str(run_dir),
        "log": str(log) if log is not None else None,
        "launcher_pid": launcher["pid"] if (alive or elsewhere) else None,
        "launcher_host": launcher["host"] if elsewhere else None,
        "exit_status": launcher["exit"],
    }


def collect(data_dir: str, arxiv_id: Optional[str] = None) -> list:
    envs = Path(data_dir) / "envs"
    if not envs.is_dir():
        return []
    papers = [envs / arxiv_id] if arxiv_id else sorted(p for p in envs.iterdir() if p.is_dir())
    rows = []
    for paper_dir in papers:
        for run in list_runs(paper_dir):
            rows.append(run_status(paper_dir.name, paper_dir / run))
    return rows


def _age(timestamp: Optional[float]) -> str:
    if timestamp is None:
        return "-"
    seconds = max(0, int(time.time() - timestamp))
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{seconds // size}{unit} ago"
    return "just now"


def cmd_status(args) -> int:
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"  {S.error(str(exc))}")
        return 1
    data_dir = cfg.get("data_dir") or ""
    arxiv_id = getattr(args, "arxiv_id", None)
    if arxiv_id:
        arxiv_id = normalize_arxiv_id(arxiv_id)
    if not data_dir:
        print(f"  {S.error(f'Data directory not set. Run {S.BOLD}sw init{S.RESET}.')}")
        return 1

    rows = collect(data_dir, arxiv_id)
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        where = f" for {arxiv_id}" if arxiv_id else ""
        print(f"\n  {S.DIM}No runs{where} under {data_dir}/envs. "
              f"Start one with: {S.RESET}sw research <arxiv_id>\n")
        return 0

    colors = {"running": S.BRIGHT_CYAN, "test phase": S.BRIGHT_MAGENTA,
              "finished": S.GREEN, "stopped": S.YELLOW, "failed": S.RED}
    def _state(row: dict) -> str:
        if row["launcher_host"]:
            return f"{row['state']} on {row['launcher_host']}"
        if row["state"] == "failed":
            return f"{row['state']} (exit {row['exit_status']})"
        return row["state"]

    # Columns as wide as their longest entry, so a long run name or host
    # never runs into the next column.
    run_width = max([len(row["run"]) for row in rows] + [3]) + 2
    state_width = max([len(_state(row)) for row in rows] + [5]) + 2
    print()
    S.table_header(("ARXIV ID", 14), ("RUN", run_width), ("STATE", state_width),
                   ("SCORES", 16), ("LAST OUTPUT", 12))
    for row in rows:
        color = colors.get(row["state"], S.DIM)
        scores = " + ".join(
            name for name, present in (("train", row["scores_train"]), ("test", row["scores_test"]))
            if present
        ) or "-"
        S.table_row(
            [row["arxiv_id"], row["run"], f"{color}{_state(row)}{S.RESET}",
             scores, f"{S.DIM}{_age(row['last_activity'])}{S.RESET}"],
            [14, run_width, state_width, 16, 12],
        )
    print()
    return 0
