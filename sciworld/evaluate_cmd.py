"""``sw evaluate``: score a run against the paper's reproduced method.

Files involved::

    <source>/scoring/reference.json     the scoring contract: experiment weights,
                                        metric directions, which methods are baselines
    <source>/scoring/scores_<split>.json  what the paper's reproduced method scored
    <run>/scoring/scores_<split>.json     what the run scored

Two views are printed. The first is local: for every scored cell (experiment,
method, metric) the paper's value, the run's value and the relative change.
The second is FNI, the cohort-normalized benchmark score (see
:mod:`sciworld.normscore`), which is the number comparable across papers.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import textwrap
from pathlib import Path
from typing import Optional

from sciworld import rundir
from sciworld import style as S
from sciworld.settings import ConfigError, load_config
from sciworld.github import DEFAULT_ORG
from sciworld.modes import MODES, canonical_mode, mode_from_run_name
from sciworld.normscore import cell_value, fni
from sciworld.papers import normalize_arxiv_id, read_source_record

_SPLITS = (("train", "Visible split"), ("test", "Held-out split"))
_NOT_RUNS = {"source", "normalization"}


# ── Runs ──────────────────────────────────────────────────────────────────


def list_runs(arxiv_dir: Path) -> list:
    """Names of the run directories of one paper, sorted."""
    if not arxiv_dir.is_dir():
        return []
    return sorted(
        p.name for p in arxiv_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name not in _NOT_RUNS
        and ((p / "status.json").exists() or (p / "scoring").is_dir())
    )


def _resolve(arxiv_id: Optional[str], run: Optional[str], path: Optional[str], data_dir: str) -> tuple:
    """Return ``(source_dir, run_dir, error)``."""
    if path:
        run_dir = Path(path).expanduser().resolve()
        if not run_dir.is_dir():
            return None, None, f"No such workspace: {run_dir}"
        recorded = rundir.read_record(run_dir).get("source")
        source = Path(recorded) if recorded and Path(recorded).is_dir() else run_dir.parent / "source"
        return source, run_dir, None
    if not data_dir:
        return None, None, f"Data directory not set. Run {S.BOLD}sw init{S.RESET} or pass {S.BOLD}--path{S.RESET}."
    arxiv_dir = Path(data_dir) / "envs" / arxiv_id
    runs = list_runs(arxiv_dir)
    if not runs:
        return None, None, (
            f"No runs found for {S.BOLD}{arxiv_id}{S.RESET} under {arxiv_dir}. "
            f"Start one with {S.BOLD}sw research {arxiv_id}{S.RESET}."
        )
    if run:
        if run not in runs:
            return None, None, f"Run {S.BOLD}{run}{S.RESET} not found. Available: {', '.join(runs)}"
        return arxiv_dir / "source", arxiv_dir / run, None
    if len(runs) == 1:
        return arxiv_dir / "source", arxiv_dir / runs[0], None
    if not S.is_interactive():
        return None, None, f"Several runs exist. Pass --run with one of: {', '.join(runs)}"
    S.bar()
    return arxiv_dir / "source", arxiv_dir / runs[S.select("Which run?", runs)], None


def _load_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):  # unreadable, not UTF-8, or not JSON
        return None
    return data if isinstance(data, dict) else None


# ── Cells ─────────────────────────────────────────────────────────────────


def _experiments(scores: Optional[dict]) -> dict:
    if not isinstance(scores, dict):
        return {}
    wrapped = scores.get("experiments")
    if isinstance(wrapped, dict):
        return wrapped
    return {k: v for k, v in scores.items() if isinstance(v, dict)}


def scored_cells(source_scores: dict, reference: Optional[dict]) -> list:
    """The cells a run is scored on, with their prior weights.

    One cell per (experiment, method, metric) the paper's reproduced method
    reports, leaving out the methods the scoring contract marks as baselines.
    A metric's direction comes from ``higher_is_better``; the magnitude of its
    coefficient times the experiment weight is the cell's weight.
    """
    ref_experiments = (reference or {}).get("experiments")
    if not isinstance(ref_experiments, dict):
        ref_experiments = {}
    cells: list = []
    for name, experiment in _experiments(source_scores).items():
        if not isinstance(experiment, dict):
            continue
        ref = ref_experiments.get(name) if isinstance(ref_experiments.get(name), dict) else {}
        exp_weight = float(ref.get("weight", 1.0) or 1.0)
        metrics = ref.get("metrics") if isinstance(ref.get("metrics"), dict) else {}
        ref_methods = ref.get("results") if isinstance(ref.get("results"), dict) else {}
        results = experiment.get("results") if isinstance(experiment.get("results"), dict) else experiment
        for method, row in results.items():
            if not isinstance(row, dict):
                continue
            spec = ref_methods.get(method)
            if isinstance(spec, dict) and spec.get("type") == "baseline":
                continue
            for metric, value in row.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                meta = metrics.get(metric) if isinstance(metrics.get(metric), dict) else {}
                coefficient = float(meta.get("coefficient", 1.0) or 0.0)
                cells.append({
                    "experiment": name,
                    "method": method,
                    "metric": metric,
                    "source": float(value),
                    "higher_is_better": bool(meta.get("higher_is_better", True)),
                    "weight": abs(coefficient) * exp_weight,
                })
    return cells


def relative_improvement(run: float, source: float, higher_is_better: bool) -> float:
    """Signed relative change; positive means the run is better.

    When the paper's value is zero the result is +1, -1 or 0 by direction.
    """
    gain = (run - source) if higher_is_better else (source - run)
    if abs(source) < 1e-12:
        return 0.0 if abs(gain) < 1e-12 else (1.0 if gain > 0 else -1.0)
    return gain / abs(source)


def compare_split(source_scores: dict, run_scores: dict, reference: Optional[dict]) -> dict:
    """Compare one split cell by cell.

    Returns ``{"rows": [...], "overall": float or None, "n_cells": int,
    "n_reported": int}``. ``overall`` is the weighted mean relative
    improvement over the cells the run reports.
    """
    cells = scored_cells(source_scores, reference)
    # Every method the paper reports for an experiment, baselines included.
    # None of them may stand in for the run's own method.
    paper_methods: dict = {}
    for name, experiment in _experiments(source_scores).items():
        if isinstance(experiment, dict):
            results = experiment.get("results") if isinstance(experiment.get("results"), dict) else experiment
            paper_methods[name] = frozenset(results)
    rows: list = []
    for cell in cells:
        others = paper_methods.get(cell["experiment"], frozenset()) - {cell["method"]}
        value = cell_value(run_scores, cell["experiment"], cell["method"], cell["metric"], others)
        row = dict(cell, run=value, relative=None)
        # A NaN or infinite score is shown, but it is not a result.
        if value is not None and math.isfinite(value):
            row["relative"] = relative_improvement(value, cell["source"], cell["higher_is_better"])
        rows.append(row)
    reported = [r for r in rows if r["relative"] is not None]
    total_weight = sum(r["weight"] for r in reported)
    overall = None
    if reported:
        if total_weight > 0:
            overall = sum(r["weight"] * r["relative"] for r in reported) / total_weight
        else:
            overall = sum(r["relative"] for r in reported) / len(reported)
    return {"rows": rows, "overall": overall, "n_cells": len(rows), "n_reported": len(reported)}


# ── Rendering ─────────────────────────────────────────────────────────────


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def _pct(value: float) -> str:
    percent = value * 100
    if abs(percent) < 0.005:
        return "0.00%"
    if abs(percent) >= 1000:
        return f"{percent:+.0f}%"
    return f"{percent:+.2f}%"


def _trend(value: float) -> tuple:
    if value > 1e-12:
        return S.GREEN, "▲"
    if value < -1e-12:
        return S.RED, "▼"
    return S.DIM, "="


def _render_split(label: str, filename: str, comparison: dict) -> None:
    S.bar()
    S.bar(f"  {S.BOLD}{label}{S.RESET}  {S.DIM}({filename}){S.RESET}")
    rows = comparison["rows"]
    if not rows:
        S.bar(f"    {S.DIM}The paper's scores list no scored cells.{S.RESET}")
        return
    width = max(len(f"{r['method']} · {r['metric']}") for r in rows)
    current = None
    for row in rows:
        if row["experiment"] != current:
            current = row["experiment"]
            S.bar(f"    {S.BOLD}{current}{S.RESET}")
        name = f"{row['method']} · {row['metric']}"
        arrow = "higher is better" if row["higher_is_better"] else "lower is better"
        if row["relative"] is None:
            missing = "not reported by the run" if row["run"] is None else f"run {row['run']} is not a number"
            S.bar(f"      {name:<{width}}  paper {_fmt(row['source'])}   {S.DIM}{missing}{S.RESET}")
            continue
        color, mark = _trend(row["relative"])
        S.bar(
            f"      {name:<{width}}  paper {_fmt(row['source'])}   run {_fmt(row['run'])}   "
            f"{color}{mark} {_pct(row['relative'])}{S.RESET}  {S.DIM}{arrow}{S.RESET}"
        )
    if comparison["overall"] is not None:
        color, mark = _trend(comparison["overall"])
        S.bar(
            f"    {S.BOLD}Weighted mean{S.RESET}  {color}{mark} {_pct(comparison['overall'])}{S.RESET}  "
            f"{S.DIM}({comparison['n_reported']}/{comparison['n_cells']} cells reported){S.RESET}"
        )
    else:
        S.bar(f"    {S.DIM}The run reports none of the scored cells.{S.RESET}")


def _render_fni(result: dict) -> None:
    S.bar()
    value = result.get("fni")
    if value is None:
        S.bar(f"  {S.BOLD}FNI{S.RESET}  {S.DIM}unavailable: {result.get('reason', 'nothing to score')}{S.RESET}")
        if "none of the scored cells" in (result.get("reason") or ""):
            S.bar(f"  {S.DIM}FNI needs the run's own method to be identifiable: report it under the "
                  f"paper's method name, or as the only row that is not one of the paper's methods.{S.RESET}")
        return
    color, mark = _trend(value)
    parts = []
    if result.get("fni_train") is not None:
        parts.append(f"visible {result['fni_train']:+.1f}")
    if result.get("fni_test") is not None:
        parts.append(f"held-out {result['fni_test']:+.1f}")
    scope = "" if result.get("complete") else f" {S.DIM}(one split only){S.RESET}"
    S.bar(
        f"  {S.BOLD}FNI{S.RESET}  {color}{mark} {value:+.1f}{S.RESET}{scope}  "
        f"{S.DIM}{' · '.join(parts)}{S.RESET}"
    )
    cohort = f" of {result['n_models']} frontier models" if result.get("n_models") else ""
    S.bar(
        f"  {S.DIM}Frontier-Normalized Improvement: 0 matches the paper's reproduced "
        f"method, 100 is the typical gain of the reference cohort{cohort}.{S.RESET}"
    )
    if result.get("coefficients_updated"):
        S.bar(
            f"  {S.DIM}Coefficients of {result['coefficients_updated']}. Compare FNI values "
            f"only when they come from the same coefficients.{S.RESET}"
        )


def _render_note(path: Path, label: str) -> None:
    """Show the scientist's ``idea.md`` or ``reflection.md`` if it was filled in."""
    try:
        raw = path.read_text()
    except OSError:
        return
    text = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL).strip()
    if not text:
        return
    width = max(60, shutil.get_terminal_size((100, 20)).columns) - 12
    lines = textwrap.wrap(" ".join(text.split()), width)
    S.bar()
    S.bar(f"  {S.BOLD}{label}{S.RESET}")
    for line in lines[:8]:
        S.bar(f"    {S.DIM}{line}{S.RESET}")
    if len(lines) > 8:
        S.bar(f"    {S.DIM}...{S.RESET}")


# ── Command ───────────────────────────────────────────────────────────────


def run_mode(run_dir: Path, mode: Optional[str] = None) -> Optional[str]:
    """The mode of a run: *mode* if given, else what ``sw research`` recorded,
    else what the directory name says. ``None`` when it cannot be told."""
    for candidate in (mode, rundir.read_record(run_dir).get("mode")):
        if candidate:
            return canonical_mode(candidate)
    return mode_from_run_name(Path(run_dir).name)


def _recorded_repository(source: Path, org: str) -> Optional[str]:
    """Name of the repository *source* was downloaded from, if it belongs to *org*.

    The coefficients are then read from that repository directly. Without the
    record, the organization is searched for the paper, which GitHub limits
    to a few requests per minute and which lags behind for new repositories.
    """
    url = str(read_source_record(source).get("clone_url") or "")
    match = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if match and match.group(1).lower() == org.lower():
        return match.group(2)
    return None


def evaluate_run(
    source: Path, run_dir: Path, arxiv_id: Optional[str], org: str = DEFAULT_ORG,
    mode: Optional[str] = None,
) -> dict:
    """Score the run at *run_dir* against the environment source at *source*.

    Returns ``{"splits": {split: comparison}, "fni": {...}, "mode": ...}``.
    A split is absent when the paper or the run has no scores for it.
    *mode* overrides the recorded mode of the run (see :func:`run_mode`).
    """
    reference = _load_json(source / "scoring" / "reference.json")
    splits: dict = {}
    run_scores: dict = {}
    for split, _label in _SPLITS:
        filename = f"scores_{split}.json"
        paper = _load_json(source / "scoring" / filename)
        mine = _load_json(run_dir / "scoring" / filename)
        run_scores[split] = mine
        if paper is not None and mine is not None:
            splits[split] = compare_split(paper, mine, reference)
    mode = run_mode(run_dir, mode)
    result = {"splits": splits, "mode": mode, "fni": None}
    if run_scores["train"] is None and run_scores["test"] is None:
        return result
    if not arxiv_id:
        result["fni"] = {"fni": None, "reason": "the paper of this run is unknown; pass its arXiv id"}
    elif mode is None:
        result["fni"] = {
            "fni": None,
            "reason": f"the mode of this run is unknown; pass --mode {' or --mode '.join(MODES)}",
        }
    else:
        local = run_dir.parent / "normalization"
        result["fni"] = fni(
            arxiv_id, mode, run_scores["train"], run_scores["test"],
            org=org, local_dir=local if local.is_dir() else None,
            repo_name=_recorded_repository(source, org),
        )
    return result


def cmd_evaluate(args) -> int:
    arxiv_id = getattr(args, "arxiv_id", None)
    run = getattr(args, "run", None)
    path = getattr(args, "path", None)
    mode = getattr(args, "mode", None)

    def _fail(message: str) -> int:
        S.bar()
        S.step_done(S.error(message))
        print()
        args._sw_error = S.plain(message)
        return 1

    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"  {S.error(str(exc))}")
        args._sw_error = str(exc)
        return 1
    org = getattr(args, "org", None) or cfg.get("org") or DEFAULT_ORG

    if arxiv_id:
        arxiv_id = normalize_arxiv_id(arxiv_id)
    if not arxiv_id and not path:
        S.intro("Scientist World", subtitle="Evaluate a run")
        return _fail(f"Usage: {S.BOLD}sw evaluate <arxiv_id> [--run NAME]{S.RESET} or {S.BOLD}--path WORKSPACE{S.RESET}")
    if mode:
        try:
            mode = canonical_mode(mode)
        except ValueError as exc:
            S.intro("Scientist World", subtitle="Evaluate a run")
            return _fail(str(exc))

    S.intro("Scientist World", subtitle=f"Evaluating {arxiv_id or path}")
    source, run_dir, error = _resolve(arxiv_id, run, path, cfg.get("data_dir") or "")
    if error:
        return _fail(error)
    if not arxiv_id:
        # --path: the paper is what sw research or the download recorded, else
        # the name of the directory that holds the run.
        for candidate in (
            rundir.read_record(run_dir).get("arxiv_id"),
            read_source_record(source).get("arxiv_id"),
            normalize_arxiv_id(run_dir.parent.name),
        ):
            if candidate and re.fullmatch(r"\d{4}\.\d{4,5}", str(candidate)):
                arxiv_id = str(candidate)
                break
    if not (source / "scoring").is_dir():
        return _fail(f"The environment source is missing: {source}")

    S.bar()
    S.step_done(f"Run:    {S.BOLD}{run_dir}{S.RESET}")
    S.step_done(f"Source: {S.BOLD}{source}{S.RESET}")
    _render_note(run_dir / "idea.md", "Idea")

    result = evaluate_run(source, run_dir, arxiv_id, org=org, mode=mode)
    if not result["splits"]:
        return _fail(
            f"No scores to compare: {run_dir.name}/scoring/ has no scores_train.json "
            "or scores_test.json that the paper also reports."
        )
    for split, label in _SPLITS:
        if split in result["splits"]:
            _render_split(label, f"scores_{split}.json", result["splits"][split])
    _render_fni(result["fni"] or {"fni": None})
    _render_note(run_dir / "reflection.md", "Reflection")
    S.step_end("")
    print()
    args._sw_result = result
    return 0
