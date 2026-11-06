"""Research modes.

``improve``   the scientist starts from the paper's method and tries to beat it.
``innovate``  the scientist sees only the problem and designs its own method.

``scratch`` is the pre-0.3 name of ``innovate`` and is still accepted.
"""

from __future__ import annotations

MODES = ("improve", "innovate")
_ALIASES = {"scratch": "innovate"}

# Accepted on the command line and in the Python API.
MODE_CHOICES = MODES + tuple(_ALIASES)


def canonical_mode(mode: str) -> str:
    """Return the canonical name of *mode*; raise ``ValueError`` if unknown."""
    name = str(mode or "").strip().lower()
    name = _ALIASES.get(name, name)
    if name not in MODES:
        raise ValueError(
            f"unknown mode {mode!r}: expected one of {', '.join(MODES)}"
        )
    return name


def mode_from_run_name(run_name: str) -> str | None:
    """Recover the mode from a run directory name such as ``codex-innovate-v2``.

    Run directories are named ``<agent>-<mode>[-<tag>]``, so the mode is the
    first mode-like segment of the name: a tag that happens to contain a mode
    word (``claude-code-improve-from-scratch``) does not change the answer.
    This is only a fallback for runs that have no record of their mode.
    """
    for part in run_name.split("-"):
        if part in MODES or part in _ALIASES:
            return canonical_mode(part)
    return None
