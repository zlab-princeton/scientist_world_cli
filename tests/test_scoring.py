"""Scoring: the cell-by-cell comparison and FNI."""

import json
import math

import pytest

from sciworld import evaluate_cmd, normscore
from sciworld.github import GitHubError


def scores(**methods):
    return {"experiments": {"exp": {"results": methods}}}


REFERENCE = {"experiments": {"exp": {
    "weight": 0.5,
    "metrics": {
        "acc": {"higher_is_better": True, "coefficient": 1.0},
        "err": {"higher_is_better": False, "coefficient": -0.5},
        "loss": {"higher_is_better": False, "coefficient": 2.0},
    },
    "results": {"paper": {"type": "proposed"}, "base": {"type": "baseline"}},
}}}


# ── Cells ──

def test_scored_cells_skip_baselines_and_non_numbers():
    source = scores(paper={"acc": 0.5, "err": 0.2, "note": "n/a", "flag": True}, base={"acc": 0.1})
    cells = evaluate_cmd.scored_cells(source, REFERENCE)
    assert [(c["method"], c["metric"]) for c in cells] == [("paper", "acc"), ("paper", "err")]
    assert cells[0]["weight"] == 0.5 and cells[1]["weight"] == 0.25


def test_direction_comes_from_higher_is_better_only():
    """The sign of a coefficient carries no meaning; its size is the weight."""
    source = scores(paper={"err": 0.2, "loss": 1.0})
    by_metric = {c["metric"]: c for c in evaluate_cmd.scored_cells(source, REFERENCE)}
    assert by_metric["err"]["higher_is_better"] is False   # negative coefficient
    assert by_metric["loss"]["higher_is_better"] is False  # positive coefficient
    assert by_metric["loss"]["weight"] == 1.0


def test_cells_without_a_contract_default_to_higher_is_better():
    cells = evaluate_cmd.scored_cells({"exp": {"m": {"acc": 1.0}}}, None)
    assert cells == [{"experiment": "exp", "method": "m", "metric": "acc", "source": 1.0,
                      "higher_is_better": True, "weight": 1.0}]


@pytest.mark.parametrize("run, source, higher, expected", [
    (0.6, 0.5, True, 0.2), (0.4, 0.5, True, -0.2), (0.4, 0.5, False, 0.2),
    (0.0, 0.0, True, 0.0), (0.1, 0.0, True, 1.0), (0.1, 0.0, False, -1.0),
    (-1.0, -2.0, True, 0.5),
])
def test_relative_improvement(run, source, higher, expected):
    assert evaluate_cmd.relative_improvement(run, source, higher) == pytest.approx(expected)


def test_compare_split_weights_and_missing_cells():
    source = scores(paper={"acc": 0.5, "err": 0.2}, base={"acc": 0.1})
    run = scores(paper={"acc": 0.75})
    result = evaluate_cmd.compare_split(source, run, REFERENCE)
    assert result["n_cells"] == 2 and result["n_reported"] == 1
    assert result["overall"] == pytest.approx(0.5)
    assert result["rows"][1]["run"] is None and result["rows"][1]["relative"] is None
    both = evaluate_cmd.compare_split(source, scores(paper={"acc": 0.75, "err": 0.1}), REFERENCE)
    # (0.5 * 0.5 + 0.25 * 0.5) / 0.75
    assert both["overall"] == pytest.approx(0.5)
    assert evaluate_cmd.compare_split(source, scores(), REFERENCE)["overall"] is None


def test_a_lone_baseline_row_is_not_taken_for_the_runs_method():
    source = scores(paper={"acc": 0.5}, base={"acc": 0.1})
    only_baseline = evaluate_cmd.compare_split(source, scores(base={"acc": 0.9}), REFERENCE)
    assert only_baseline["n_reported"] == 0 and only_baseline["overall"] is None
    with_own = evaluate_cmd.compare_split(source, scores(base={"acc": 0.9}, mine={"acc": 0.6}), REFERENCE)
    assert with_own["rows"][0]["run"] == 0.6


# ── Finding the run's row ──

def test_exact_method_name_wins():
    run = scores(paper={"acc": 0.9}, mine={"acc": 0.1})
    assert normscore.cell_value(run, "exp", "paper", "acc") == 0.9


def test_a_single_differently_named_row_is_the_runs_method():
    run = scores(my_new_method={"acc": 0.8})
    assert normscore.cell_value(run, "exp", "paper", "acc") == 0.8


def test_the_papers_other_methods_are_never_borrowed():
    run = scores(base={"acc": 0.3})
    assert normscore.cell_value(run, "exp", "paper", "acc", frozenset({"base"})) is None
    run = scores(base={"acc": 0.3}, mine={"acc": 0.8})
    assert normscore.cell_value(run, "exp", "paper", "acc", frozenset({"base"})) == 0.8


def test_ambiguous_rows_are_not_guessed():
    run = scores(first={"acc": 0.8}, second={"acc": 0.7})
    assert normscore.cell_value(run, "exp", "paper", "acc") is None


@pytest.mark.parametrize("run", [
    {}, {"exp": "text"}, scores(paper={"acc": "high"}), scores(paper={"acc": True}),
    {"experiments": {"other": {"results": {"paper": {"acc": 1}}}}},
])
def test_unusable_values_are_absent(run):
    assert normscore.cell_value(run, "exp", "paper", "acc") is None


def test_flat_layout_is_read_too():
    assert normscore.cell_value({"exp": {"paper": {"acc": 0.4}}}, "exp", "paper", "acc") == 0.4


# ── normalize ──

def coeffs(*cells, **extra):
    return dict({"n_models": 4, "cells": list(cells)}, **extra)


def cell(metric="acc", source=0.5, higher=True, coefficient=2.0, method="paper"):
    return {"experiment": "exp", "method": method, "metric": metric, "source": source,
            "higher_is_better": higher, "coefficient": coefficient}


def test_normalize_sums_coefficient_times_signed_gain():
    c = coeffs(cell("acc", 0.5, True, 2.0), cell("err", 0.2, False, 10.0))
    result = normscore.normalize(c, scores(paper={"acc": 0.75, "err": 0.25}))
    assert result["score"] == pytest.approx(2.0 * 0.25 + 10.0 * -0.05)
    assert result["n_cells_used"] == 2 and result["n_cells_missing"] == 0 and result["n_models"] == 4


def test_normalize_skips_pruned_missing_and_broken_cells():
    c = coeffs(
        cell("acc"), cell("err", 0.2, False, 10.0), cell("loss", 1.0, False, 3.0),
        cell("nan", 0.0, True, float("nan")), cell("none", 0.0, True, None),
        pruned_cells=[{"experiment": "exp", "method": "paper", "metric": "err"}],
    )
    result = normscore.normalize(c, scores(paper={"acc": 0.75, "err": 0.0, "nan": 1.0, "none": 1.0}))
    assert result["score"] == pytest.approx(0.5)
    assert (result["n_cells_used"], result["n_cells_missing"], result["n_cells_pruned"]) == (1, 1, 1)


@pytest.mark.parametrize("c, s, reason", [
    (None, scores(paper={"acc": 1}), "no coefficients"),
    ({"excluded": True, "reason": "cohort_too_small", "n_models": 2}, scores(paper={"acc": 1}),
     "not in the scored pool (too few reference models have run it)"),
    ({"excluded": True, "reason": "cohort_too_small", "n_models": 0, "min_required": 3}, scores(paper={"acc": 1}),
     "too few reference models have run it, 0 of the 3 needed"),
    ({"excluded": True, "reason": "all_cells_tied", "n_models": 5}, scores(paper={"acc": 1}),
     "the reference models tie on every cell"),
    ({"excluded": True, "reason": "a_reason_added_later"}, scores(paper={"acc": 1}), "a_reason_added_later"),
    ({"excluded": True}, scores(paper={"acc": 1}), "it has no usable scale"),
    (coeffs(cell()), None, "no scores"),
    (coeffs(), scores(paper={"acc": 1}), "no cells"),
    (coeffs(cell()), scores(), "none of the scored cells"),
])
def test_normalize_explains_why_there_is_no_score(c, s, reason):
    result = normscore.normalize(c, s)
    assert result["score"] is None and reason in result["reason"]


def test_an_excluded_paper_keeps_its_reason_code_for_scripts():
    result = normscore.normalize({"excluded": True, "reason": "cohort_too_small", "n_models": 1}, scores(paper={"acc": 1}))
    assert result["excluded"] == "cohort_too_small" and result["n_models"] == 1
    assert "excluded" not in normscore.normalize(coeffs(cell()), scores(paper={"acc": 0.5}))


def test_matching_the_paper_scores_zero():
    assert normscore.normalize(coeffs(cell()), scores(paper={"acc": 0.5}))["score"] == 0.0


# ── FNI ──

def write_coeffs(directory, mode, **by_split):
    directory.mkdir(parents=True, exist_ok=True)
    for split, data in by_split.items():
        (directory / f"coeffs_{mode}_{split}.json").write_text(json.dumps(data))


@pytest.fixture
def offline(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("GitHub must not be contacted")
    monkeypatch.setattr(normscore, "find_environment", refuse)
    monkeypatch.setattr(normscore, "fetch_file", refuse)


def test_fni_is_the_mean_of_both_splits_times_100(tmp_path, offline):
    write_coeffs(tmp_path, "improve", train=coeffs(cell(source=0.5)), test=coeffs(cell(source=0.6)))
    result = normscore.fni("2401.00001", "improve", scores(paper={"acc": 1.0}), scores(paper={"acc": 1.0}),
                           local_dir=tmp_path)
    assert result["fni_train"] == pytest.approx(100.0) and result["fni_test"] == pytest.approx(80.0)
    assert result["fni"] == pytest.approx(90.0) and result["complete"] is True and result["n_models"] == 4


def test_fni_names_the_day_its_coefficients_were_computed(tmp_path, offline):
    """The newer of the two files, so a stale set is visible next to the number."""
    write_coeffs(tmp_path, "improve",
                 train=dict(coeffs(cell()), last_updated="2026-07-13T05:13:21.814922+00:00"),
                 test=dict(coeffs(cell()), last_updated="2026-09-18T20:28:34+00:00"))
    run = scores(paper={"acc": 0.7})
    assert normscore.fni("x", "improve", run, run, local_dir=tmp_path)["coefficients_updated"] == "2026-09-18"

    write_coeffs(tmp_path, "innovate", train=dict(coeffs(cell()), last_updated="not a date"))
    assert normscore.fni("x", "innovate", run, None, local_dir=tmp_path)["coefficients_updated"] is None


def test_fni_with_one_split_is_marked_incomplete(tmp_path, offline):
    write_coeffs(tmp_path, "innovate", train=coeffs(cell(source=0.5)), test=coeffs(cell(source=0.6)))
    result = normscore.fni("2401.00001", "scratch", scores(mine={"acc": 0.4}), None, local_dir=tmp_path)
    assert result["fni"] == pytest.approx(-20.0) and result["fni_test"] is None
    assert result["complete"] is False and result["mode"] == "innovate"


def test_fni_uses_the_coefficients_of_the_runs_mode(tmp_path, offline):
    write_coeffs(tmp_path, "improve", train=coeffs(cell(coefficient=1.0)))
    write_coeffs(tmp_path, "innovate", train=coeffs(cell(coefficient=5.0)))
    run = scores(paper={"acc": 0.7})
    assert normscore.fni("x", "improve", run, None, local_dir=tmp_path)["fni"] == pytest.approx(20.0)
    assert normscore.fni("x", "innovate", run, None, local_dir=tmp_path)["fni"] == pytest.approx(100.0)


def test_fni_reports_unknown_modes_and_unscorable_runs(tmp_path, offline):
    assert "unknown mode" in normscore.fni("x", None, {}, {})["reason"]
    write_coeffs(tmp_path, "improve", train={"excluded": True, "reason": "cohort_too_small"})
    result = normscore.fni("x", "improve", scores(paper={"acc": 1}), None, local_dir=tmp_path)
    assert result["fni"] is None and "not in the scored pool" in result["reason"]
    assert result["details"]["train"]["excluded"] == "cohort_too_small"


def test_coefficients_are_fetched_from_the_normalization_branch(monkeypatch):
    from sciworld.github import RepoInfo
    requested = []

    def fake_find(arxiv_id, org, token):
        return RepoInfo(arxiv_id, "full", "T", f"{arxiv_id}-repo", "url")

    def fake_fetch(repo_name, path, ref=None, org=None, token=None):
        requested.append((repo_name, path, ref, org))
        if path.endswith("_train.json"):
            return json.dumps(coeffs(cell())).encode()
        return None

    monkeypatch.setattr(normscore, "find_environment", fake_find)
    monkeypatch.setattr(normscore, "fetch_file", fake_fetch)
    result = normscore.fni("2401.00001", "improve", scores(paper={"acc": 0.75}), None, org="Org")
    assert result["fni"] == pytest.approx(50.0)
    assert requested == [
        ("2401.00001-repo", "normalization/coeffs_improve_train.json", "normalization", "Org"),
        ("2401.00001-repo", "normalization/coeffs_improve_test.json", "normalization", "Org"),
    ]


def test_fni_without_published_coefficients_or_network(monkeypatch):
    from sciworld.github import RepoInfo
    monkeypatch.setattr(normscore, "find_environment", lambda *a, **k: RepoInfo("x", "full", "T", "r", "u"))
    monkeypatch.setattr(normscore, "fetch_file", lambda *a, **k: None)
    assert "no normalization coefficients" in normscore.fni("x", "improve", {}, {})["reason"]

    def offline(*args, **kwargs):
        raise GitHubError("cannot reach api.github.com")
    monkeypatch.setattr(normscore, "find_environment", offline)
    assert "cannot reach" in normscore.fni("x", "improve", {}, {})["reason"]
    monkeypatch.setattr(normscore, "find_environment", lambda *a, **k: None)
    assert "no environment repository" in normscore.fni("x", "improve", {}, {})["reason"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_score_that_is_not_finite_is_reported_as_unscorable(tmp_path, offline, bad):
    write_coeffs(tmp_path, "improve", train=coeffs(cell()), test=coeffs(cell()))
    result = normscore.fni("x", "improve", scores(paper={"acc": bad}), scores(paper={"acc": 0.75}),
                           local_dir=tmp_path)
    assert result["fni_train"] is None and result["fni"] == pytest.approx(50.0) and not result["complete"]
    assert "not a finite number" in result["details"]["train"]["reason"]


def test_a_score_that_is_not_finite_is_shown_but_not_counted():
    result = evaluate_cmd.compare_split(
        scores(paper={"acc": 0.5, "f1": 0.5}), scores(paper={"acc": float("nan"), "f1": 0.75}), None)
    by_metric = {row["metric"]: row for row in result["rows"]}
    assert by_metric["acc"]["relative"] is None and math.isnan(by_metric["acc"]["run"])
    assert result["n_reported"] == 1 and result["overall"] == pytest.approx(0.5)


def test_a_failed_download_is_not_reported_as_no_coefficients(monkeypatch):
    """A rate-limited request used to read "no normalization coefficients are
    published for this paper"."""
    from sciworld.github import RepoInfo
    monkeypatch.setattr(normscore, "find_environment", lambda *a, **k: RepoInfo("x", "full", "T", "r", "u"))

    def rate_limited(*args, **kwargs):
        raise GitHubError("Reading normalization/coeffs_improve_train.json of Org/r: GitHub API rate limit reached.")
    monkeypatch.setattr(normscore, "fetch_file", rate_limited)
    reason = normscore.fni("x", "improve", scores(paper={"acc": 1}), None)["reason"]
    assert "rate limit" in reason and "no normalization coefficients" not in reason
    with pytest.raises(GitHubError):
        normscore.load_coefficients("x", "improve")


def test_the_mode_of_a_run(tmp_path):
    from sciworld import rundir
    run = tmp_path / "claude-code-improve-from-scratch"
    run.mkdir()
    assert evaluate_cmd.run_mode(run) == "improve"                 # the name: first mode word
    assert evaluate_cmd.run_mode(run, "scratch") == "innovate"     # stated explicitly
    rundir.write_record(run, {"mode": "innovate"})
    assert evaluate_cmd.run_mode(run) == "innovate"                # recorded by sw research
    assert evaluate_cmd.run_mode(tmp_path / "my-own-folder") is None


def test_evaluate_run_says_what_is_missing_instead_of_guessing(tmp_path, offline):
    source, run = tmp_path / "source", tmp_path / "trial"
    (source / "scoring").mkdir(parents=True)
    (run / "scoring").mkdir(parents=True)
    for directory in (source, run):
        (directory / "scoring" / "scores_train.json").write_text(json.dumps(scores(paper={"acc": 0.5})))
    result = evaluate_cmd.evaluate_run(source, run, "2401.00001")
    assert result["mode"] is None and "--mode improve or --mode innovate" in result["fni"]["reason"]
    assert "arXiv id" in evaluate_cmd.evaluate_run(source, run, None, mode="improve")["fni"]["reason"]
    write_coeffs(tmp_path / "normalization", "improve", train=coeffs(cell()))
    stated = evaluate_cmd.evaluate_run(source, run, "2401.00001", mode="improve")
    assert stated["mode"] == "improve" and stated["fni"]["fni_train"] is not None


def test_evaluate_run_reads_coefficients_from_the_repository_it_downloaded(tmp_path, monkeypatch):
    """The source records where it came from, so nothing has to be searched."""
    source, run = tmp_path / "source", tmp_path / "claude-code-improve"
    for directory in (source, run):
        (directory / "scoring").mkdir(parents=True)
        (directory / "scoring" / "scores_train.json").write_text(json.dumps(scores(paper={"acc": 0.75})))
    (source / ".sciworld.json").write_text(json.dumps(
        {"arxiv_id": "2401.00001", "clone_url": "https://github.com/Org/2401.00001-a-paper.git"}))
    requested = []

    def no_search(*args, **kwargs):
        raise AssertionError("the organization must not be searched")

    def fake_fetch(repo_name, path, ref=None, org=None, token=None):
        requested.append((repo_name, org))
        return json.dumps(coeffs(cell())).encode() if path.endswith("_train.json") else None

    monkeypatch.setattr(normscore, "find_environment", no_search)
    monkeypatch.setattr(normscore, "fetch_file", fake_fetch)
    result = evaluate_cmd.evaluate_run(source, run, "2401.00001", org="Org")
    assert result["fni"]["fni"] == pytest.approx(50.0)
    assert set(requested) == {("2401.00001-a-paper", "Org")}

    # A source of another organization, or without a record, is looked up.
    assert evaluate_cmd._recorded_repository(source, "Other") is None
    (source / ".sciworld.json").unlink()
    assert evaluate_cmd._recorded_repository(source, "Org") is None


@pytest.mark.parametrize("content", [b"\xff\xfe\x00", b"{broken", b"[1, 2]"])
def test_unreadable_score_files_are_treated_as_absent(tmp_path, content):
    path = tmp_path / "scores_train.json"
    path.write_bytes(content)
    assert evaluate_cmd._load_json(path) is None


def test_scale_is_100():
    assert normscore.FNI_SCALE == 100.0 and not math.isnan(normscore.FNI_SCALE)


# ── Run discovery ──

def test_list_runs_ignores_everything_that_is_not_a_run(tmp_path):
    paper = tmp_path / "2401.00001"
    for name in ("source", "normalization", ".sciworld", "claude-code-improve", "codex-innovate", "notes"):
        (paper / name).mkdir(parents=True)
    (paper / "claude-code-improve" / "status.json").write_text("{}")
    (paper / "codex-innovate" / "scoring").mkdir()
    (paper / "source" / "scoring").mkdir()
    assert evaluate_cmd.list_runs(paper) == ["claude-code-improve", "codex-innovate"]
    assert evaluate_cmd.list_runs(tmp_path / "missing") == []
