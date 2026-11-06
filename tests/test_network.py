"""Checks against github.com. Skipped unless SCIWORLD_TEST_NETWORK=1.

They use the two public environments of the ScientistWorld organization and
no token, so they exercise the anonymous path a first-time user takes.
"""

import pytest

from sciworld import github
from sciworld.papers import clone_environment

pytestmark = pytest.mark.network

PUBLIC_PAPER = "2502.13759"


def test_anonymous_listing_shows_public_environments():
    repos = github.list_environments(github.DEFAULT_ORG)
    ids = {r.arxiv_id for r in repos}
    assert PUBLIC_PAPER in ids
    assert all(not r.private and "done" in r.topics for r in repos)


def test_anonymous_lookup_and_clone(tmp_path):
    info = github.find_environment(PUBLIC_PAPER)
    assert info is not None and info.repo_name.startswith(PUBLIC_PAPER) and info.title
    source = tmp_path / "source"
    clone_environment(info.clone_url, source)
    assert (source / "briefing" / "problem.md").is_file()
    assert (source / "eval" / "train").is_dir() and (source / "eval" / "test").is_dir()
    assert (source / "scripts" / "evaluate_train.sh").is_file()
    assert not (source / ".git").exists()


def test_an_environment_that_does_not_exist_is_not_found():
    assert github.find_environment("0001.00001") is None


def test_coefficient_lookup_distinguishes_absent_from_unreadable():
    """The contents API answers 404 for a file or branch that is not there;
    that must read as "none published", never as an error."""
    info = github.find_environment(PUBLIC_PAPER)
    assert github.fetch_file(info.repo_name, "normalization/no_such_file.json", ref="no-such-branch") is None
    readme = github.fetch_file(info.repo_name, "README.md")
    assert readme and b"\n" in readme  # raw bytes, not a JSON envelope
