"""Cache of the environment listing, so browsing does not hit GitHub each time."""

from __future__ import annotations

import json
import time
from pathlib import Path

from sciworld.settings import config_dir


def cache_path(org: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in org)
    return config_dir() / "cache" / f"{safe}_repos.json"


def load_cached(org: str, ttl: int = 86400, authenticated: bool = False):
    """Return the cached listing, or ``None`` if missing, stale or unusable.

    A listing fetched without a token only contains public environments, so
    it is not reused once a token is available.
    """
    path = cache_path(org)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("repos"), list):
        return None
    if time.time() - float(data.get("timestamp", 0) or 0) > ttl:
        return None
    if authenticated and not data.get("authenticated", False):
        return None
    return data["repos"]


def save_cached(org: str, repos: list, authenticated: bool = False) -> None:
    path = cache_path(org)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"timestamp": time.time(), "authenticated": authenticated, "repos": repos}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def invalidate(org: str) -> None:
    path = cache_path(org)
    if path.exists():
        path.unlink()
