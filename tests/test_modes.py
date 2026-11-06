import pytest

from sciworld.modes import MODES, canonical_mode, mode_from_run_name


def test_modes_are_improve_and_innovate():
    assert MODES == ("improve", "innovate")


@pytest.mark.parametrize("given, expected", [
    ("improve", "improve"), ("innovate", "innovate"), ("scratch", "innovate"),
    (" Innovate ", "innovate"), ("SCRATCH", "innovate"),
])
def test_canonical_mode(given, expected):
    assert canonical_mode(given) == expected


@pytest.mark.parametrize("bad", ["", None, "reproduce", "improve2"])
def test_canonical_mode_rejects_unknown(bad):
    with pytest.raises(ValueError):
        canonical_mode(bad)


@pytest.mark.parametrize("run, expected", [
    ("claude-code-improve", "improve"),
    ("codex-innovate-v2", "innovate"),
    ("codex-scratch", "innovate"),
    ("gpt-5-xhigh-innovate-eval", "innovate"),
    ("my-own-folder", None),
    # A tag that contains a mode word does not change the mode of the run.
    ("claude-code-improve-from-scratch", "improve"),
    ("codex-innovate-improve-lr", "innovate"),
    ("codex-innovate-no-improve", "innovate"),
])
def test_mode_from_run_name(run, expected):
    assert mode_from_run_name(run) == expected
