"""Names of the tmux/screen sessions that hold runs.

A run of paper ``2505.19252`` in the directory ``codex-innovate-v2`` lives in
the session ``sw_2505_19252_codex-innovate-v2``. A workspace built with
``--path`` gets a short suffix derived from its location, so two workspaces
with the same directory name do not share a session name.
"""

from __future__ import annotations

import hashlib
import re

_PREFIX = "sw"


def session_name(arxiv_id: str, run_name: str, location: str = "") -> str:
    """Session name of a run; safe for tmux, screen and the shell.

    *location* is the workspace path of a run outside the data directory.
    """
    name = f"{_PREFIX}_{arxiv_id.replace('.', '_')}_{run_name}"
    name = re.sub(r"[^A-Za-z0-9_\-]", "_", name)
    if location:
        name += "_" + hashlib.sha1(location.encode()).hexdigest()[:6]
    return name


def parse_session_name(name: str):
    """Inverse of :func:`session_name` for runs under the data directory:
    ``(arxiv_id, run_name)`` or ``None``."""
    match = re.match(rf"^{_PREFIX}_(\d{{4}})_(\d{{4,5}})_(.+)$", name)
    if not match:
        return None
    return f"{match.group(1)}.{match.group(2)}", match.group(3)
