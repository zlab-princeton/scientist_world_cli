"""Frontier-Normalized Improvement (FNI): the benchmark score of a run.

``sw evaluate`` first shows a local, per-metric comparison with the paper's
reproduced method. The number that is comparable across papers and with the
published leaderboards is FNI::

    FNI = 100 * sum over cells of  coefficient * improvement

A cell is one (experiment, method, metric). ``improvement`` is the signed gain
of the run over the paper's reproduced method on that cell, and the
coefficient folds in the cell's weight and the spread of a frozen reference
cohort of frontier models. 0 means "matches the paper", a positive value
improves on it, and 100 is the typical gain of the reference cohort.

Coefficients are frozen per paper, mode and split, and published on the
``normalization`` branch of the environment's repository as
``normalization/coeffs_<mode>_<split>.json``. The visible (train) and held-out
(test) splits are scored separately; the leaderboard number is their mean.

This module only reads coefficients. It is a stdlib-only port of the scoring
step of the evaluation harness.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Optional

from sciworld.github import DEFAULT_ORG, GitHubError, fetch_file, find_environment, github_token
from sciworld.modes import canonical_mode

FNI_SCALE = 100.0
COEFF_BRANCH = "normalization"
SPLITS = ("train", "test")

# A paper outside the scored pool still has coefficient files; they are
# flagged ``excluded`` with one of these reasons.
EXCLUSIONS = {
    "cohort_too_small": "too few reference models have run it",
    "no_cells": "no reference model reported one of its scored cells",
    "all_cells_tied": "the reference models tie on every cell",
    "all_priors_zero": "every scored cell has zero weight",
    "paper_baseline_zero": "the reference models all match the paper",
}


# ── Scoring ───────────────────────────────────────────────────────────────


def _experiments(scores: Optional[dict]) -> dict:
    """Experiments of a scores file, with or without the ``experiments`` wrapper."""
    if not isinstance(scores, dict):
        return {}
    wrapped = scores.get("experiments")
    if isinstance(wrapped, dict):
        return wrapped
    return {k: v for k, v in scores.items() if isinstance(v, dict)}


def _known_methods(cells: list, current: dict) -> frozenset:
    """The paper's other method names in the same experiment."""
    experiment = current.get("experiment")
    return frozenset(
        c.get("method") for c in (cells or [])
        if c.get("experiment") == experiment and c.get("method") != current.get("method")
    )


def cell_value(
    scores: dict, experiment: str, method: str, metric: str,
    known_methods: frozenset = frozenset(),
) -> Optional[float]:
    """The run's value for one cell, or ``None`` if it reported none.

    A scientist's own method rarely carries the paper's method name. When
    the exact name is missing, the run's row is used if it is unambiguous:
    exactly one other row reports the metric and it is not one of the paper's
    other methods (a baseline, for example).
    """
    entry = _experiments(scores).get(experiment)
    if not isinstance(entry, dict):
        return None
    results = entry.get("results") if isinstance(entry.get("results"), dict) else entry
    if not isinstance(results, dict):
        return None

    def _number(row: dict) -> Optional[float]:
        value = row.get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    row = results.get(method)
    if isinstance(row, dict):
        value = _number(row)
        if value is not None:
            return value
    candidates = [
        name for name, other in results.items()
        if isinstance(other, dict) and name != method
        and name not in known_methods and _number(other) is not None
    ]
    if len(candidates) == 1:
        return _number(results[candidates[0]])
    return None


def normalize(coeffs: Optional[dict], scores: Optional[dict]) -> dict:
    """Score one split of a run against a paper's frozen coefficients.

    Returns ``{"score", "n_cells_used", "n_cells_missing", "n_cells_pruned",
    "n_models"}``. ``score`` is on the stored scale (multiply by
    :data:`FNI_SCALE` for FNI) and is ``None`` with a ``reason`` when the
    split cannot be scored. For a paper outside the scored pool the result
    also carries ``excluded``, the reason code of the coefficient file (a key
    of :data:`EXCLUSIONS`).
    """
    if not coeffs:
        return {"score": None, "reason": "no coefficients"}
    n_models = int(coeffs.get("n_models", 0) or 0)
    if coeffs.get("excluded"):
        code = str(coeffs.get("reason") or "")
        why = EXCLUSIONS.get(code, code or "it has no usable scale")
        needed = coeffs.get("min_required")
        if code == "cohort_too_small" and isinstance(needed, int) and not isinstance(needed, bool):
            why += f", {n_models} of the {needed} needed"
        return {"score": None, "n_models": n_models, "excluded": code or "excluded",
                "reason": f"the paper is not in the scored pool ({why})"}
    if not isinstance(scores, dict):
        return {"score": None, "n_models": n_models, "reason": "no scores"}
    cells = coeffs.get("cells") or []
    if not cells:
        return {"score": None, "n_models": n_models, "reason": "no cells"}

    pruned = {
        (p.get("experiment"), p.get("method"), p.get("metric"))
        for p in (coeffs.get("pruned_cells") or [])
    }
    total = 0.0
    used = missing = skipped = 0
    for cell in cells:
        if (cell.get("experiment"), cell.get("method"), cell.get("metric")) in pruned:
            skipped += 1
            continue
        coefficient = cell.get("coefficient")
        if not isinstance(coefficient, (int, float)) or math.isnan(coefficient):
            continue
        value = cell_value(
            scores, cell.get("experiment", ""), cell.get("method", ""),
            cell.get("metric", ""), known_methods=_known_methods(cells, cell),
        )
        if value is None:
            missing += 1
            continue
        source = float(cell.get("source", 0.0) or 0.0)
        higher = bool(cell.get("higher_is_better", True))
        total += coefficient * ((value - source) if higher else (source - value))
        used += 1

    result = {"n_cells_used": used, "n_cells_missing": missing,
              "n_cells_pruned": skipped, "n_models": n_models}
    if used == 0:
        result.update(score=None, reason="the run reports none of the scored cells")
    else:
        result["score"] = total
    return result


# ── Coefficients ──────────────────────────────────────────────────────────


def _computed_on(coeffs: Optional[dict]) -> Optional[str]:
    """The day a coefficient file was computed, as ``YYYY-MM-DD``."""
    stamp = str((coeffs or {}).get("last_updated") or "")
    return stamp[:10] if re.match(r"\d{4}-\d{2}-\d{2}", stamp) else None


def _load_local(directory: Path, mode: str, split: str) -> Optional[dict]:
    try:
        data = json.loads((Path(directory) / f"coeffs_{mode}_{split}.json").read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def load_coefficients(
    arxiv_id: str, mode: str, *,
    org: str = DEFAULT_ORG, local_dir: Optional[Path] = None,
    repo_name: Optional[str] = None,
) -> dict:
    """Return ``{"train": coeffs_or_None, "test": coeffs_or_None}``.

    *local_dir* (a directory holding ``coeffs_<mode>_<split>.json``) is used
    when it has the files; otherwise they are fetched from the
    ``normalization`` branch of the paper's repository. A split is ``None``
    only when the repository really has no file for it. Raises
    :class:`GitHubError` when GitHub is needed and cannot be reached or
    refuses the request (rate limit, access).
    """
    mode = canonical_mode(mode)
    found = {split: None for split in SPLITS}
    if local_dir is not None:
        found = {split: _load_local(local_dir, mode, split) for split in SPLITS}
        if any(found.values()):
            return found

    token = github_token()
    if not repo_name:
        info = find_environment(arxiv_id, org=org, token=token)
        if info is None:
            raise GitHubError(f"no environment repository found for {arxiv_id} in {org}")
        repo_name = info.repo_name
    for split in SPLITS:
        raw = fetch_file(
            repo_name, f"{COEFF_BRANCH}/coeffs_{mode}_{split}.json",
            ref=COEFF_BRANCH, org=org, token=token,
        )
        if raw is None:
            continue
        try:
            data = json.loads(raw)
        except ValueError:
            continue
        if isinstance(data, dict):
            found[split] = data
    return found


def fni(
    arxiv_id: str, mode: Optional[str],
    train_scores: Optional[dict], test_scores: Optional[dict], *,
    org: str = DEFAULT_ORG, local_dir: Optional[Path] = None,
    repo_name: Optional[str] = None,
) -> dict:
    """FNI of a run.

    Returns a dict with ``fni`` (the mean of the splits that could be
    scored), ``fni_train``, ``fni_test``, ``complete`` (both splits scored),
    ``n_models`` (size of the reference cohort), ``coefficients_updated``
    (the day the coefficients were computed) and per-split ``details``.
    ``fni`` is ``None`` with a ``reason`` when nothing could be scored.

    Two FNI values are comparable only when they were computed with the same
    coefficients: compare ``coefficients_updated`` along with them.
    """
    try:
        mode = canonical_mode(mode or "")
    except ValueError:
        return {"fni": None, "reason": f"unknown mode {mode!r}"}
    try:
        coeffs = load_coefficients(
            arxiv_id, mode, org=org, local_dir=local_dir, repo_name=repo_name,
        )
    except GitHubError as exc:
        return {"fni": None, "reason": str(exc)}
    if not any(coeffs.values()):
        return {"fni": None, "mode": mode,
                "reason": "no normalization coefficients are published for this paper and mode"}

    details = {
        "train": normalize(coeffs["train"], train_scores),
        "test": normalize(coeffs["test"], test_scores),
    }
    scaled = {}
    for split, detail in details.items():
        score = detail.get("score")
        if score is not None and not math.isfinite(score):
            # A NaN or an infinity in the run's scores makes the whole split
            # meaningless.
            detail["reason"] = "the run reports a value that is not a finite number"
            score = None
        scaled[split] = None if score is None else FNI_SCALE * score
    available = [v for v in scaled.values() if v is not None]
    n_models = max((d.get("n_models") or 0) for d in details.values())
    computed = sorted(day for day in (_computed_on(coeffs[split]) for split in SPLITS) if day)
    result = {
        "fni": sum(available) / len(available) if available else None,
        "fni_train": scaled["train"],
        "fni_test": scaled["test"],
        "complete": len(available) == len(SPLITS),
        "mode": mode,
        "n_models": n_models,
        "coefficients_updated": computed[-1] if computed else None,
        "details": details,
    }
    if not available:
        reasons = {d.get("reason") for d in details.values() if d.get("reason")}
        result["reason"] = "; ".join(sorted(reasons)) or "nothing to score"
    return result
