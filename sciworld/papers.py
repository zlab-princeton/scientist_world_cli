"""The catalog of environments: cached listing and single-paper lookup."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from sciworld import style as S
from sciworld.cache import load_cached, save_cached
from sciworld.fsutil import remove_tree, sweep_stale_temp
from sciworld.settings import load_config
from sciworld.github import (
    GitHubError, clone_repo, find_environment, github_token, list_environments,
)

_ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}$")


def normalize_arxiv_id(value: str) -> str:
    """Accept ``2505.19252``, ``arXiv:2505.19252v2`` or an arxiv.org URL."""
    text = (value or "").strip()
    match = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", text)
    return match.group(1) if match else text


def is_arxiv_id(value: str) -> bool:
    return bool(_ARXIV_ID.match(value or ""))


def fetch_papers(org: str, refresh: bool = False, quiet: bool = False) -> Optional[list]:
    """Return every finished environment of *org* as a list of dicts.

    Uses the cache unless *refresh* is set. Prints progress unless *quiet*.
    Returns ``None`` after printing the reason when the listing fails.
    """
    cfg = load_config()
    ttl = int(cfg.get("cache_ttl", 86400) or 0)
    token = github_token()

    if not refresh:
        cached = load_cached(org, ttl=ttl, authenticated=bool(token))
        if cached is not None:
            if not quiet:
                S.step_done(f"Loaded {S.BOLD}{len(cached)}{S.RESET} environments {S.DIM}(cached){S.RESET}")
            return cached

    live = S.is_interactive() and not quiet
    if not quiet:
        S.bar()
        S.step_start("Fetching environments")

    def _progress(fetched: int) -> None:
        if live:
            sys.stdout.write(f"{S.MOVE_UP}{S.CLEAR_LINE}")
            S.step_start(f"Fetching environments  {S.DIM}{fetched} so far{S.RESET}")

    try:
        repos = list_environments(org=org, token=token, on_progress=_progress)
    except GitHubError as exc:
        if not quiet:
            S.bar()
            S.step_done(S.error(str(exc)))
        return None

    papers = [r.to_dict() for r in repos]
    save_cached(org, papers, authenticated=bool(token))
    if not quiet:
        if live:
            sys.stdout.write(f"{S.MOVE_UP}{S.CLEAR_LINE}")
        S.step_done(f"Loaded {S.BOLD}{len(papers)}{S.RESET} environments")
        if not token:
            S.note(["No GitHub token found: only public environments are listed."])
    return papers


def find_paper(papers: list, arxiv_id: str) -> Optional[dict]:
    """Exact match on the arXiv id, else a unique prefix match."""
    for paper in papers:
        if paper.get("arxiv_id") == arxiv_id:
            return paper
    matches = [p for p in papers if p.get("arxiv_id", "").startswith(arxiv_id)]
    return matches[0] if len(matches) == 1 else None


def lookup_paper(arxiv_id: str, org: str) -> Optional[dict]:
    """Find one environment: the cache first, then a targeted GitHub query.

    Returns ``None`` if the organization has no environment for the paper.
    Raises :class:`GitHubError` if GitHub cannot be queried.
    """
    cfg = load_config()
    token = github_token()
    cached = load_cached(org, ttl=int(cfg.get("cache_ttl", 86400) or 0), authenticated=bool(token))
    if cached is not None:
        found = find_paper(cached, arxiv_id)
        if found:
            return found
    info = find_environment(arxiv_id, org=org, token=token)
    return info.to_dict() if info else None


# ── Downloaded environments ───────────────────────────────────────────────

# Written into a downloaded environment: which paper it is and what was cloned.
SOURCE_RECORD = ".sciworld.json"


def is_environment(path: Path) -> bool:
    """True if *path* looks like a downloaded environment."""
    return (Path(path) / "briefing" / "problem.md").is_file()


def read_source_record(source_dir: Path) -> dict:
    """The record written when *source_dir* was cloned, or ``{}``."""
    try:
        data = json.loads((Path(source_dir) / SOURCE_RECORD).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def check_source(source_dir: Path, arxiv_id: str) -> Optional[str]:
    """Why *source_dir* cannot serve as the environment of *arxiv_id*, or
    ``None`` if it can.

    A directory that is not an environment, or that was downloaded for
    another paper, must never be packaged under this paper's name.
    """
    source_dir = Path(source_dir)
    if not is_environment(source_dir):
        return (
            f"{source_dir} exists and is not a Scientist World environment "
            "(it has no briefing/problem.md)."
        )
    recorded = read_source_record(source_dir).get("arxiv_id")
    if recorded and arxiv_id and recorded != arxiv_id:
        return f"{source_dir} holds the environment of {recorded}, not of {arxiv_id}."
    return None


def _head_commit(repo: Path) -> str:
    if not shutil.which("git"):
        return ""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def clone_environment(
    clone_url: str, source_dir: Path, *, arxiv_id: str = "", replace: bool = False,
) -> None:
    """Clone an environment repository to *source_dir*.

    The clone goes to a temporary sibling first and is moved into place when
    complete, so an interrupted or failed clone never leaves a half-filled
    directory behind and never destroys what was there. With *replace* an
    existing *source_dir* is swapped for the new clone; without it an
    existing one is kept. ``.git`` is removed (the source is a read-only
    reference, not a checkout to push from) and the origin is recorded in
    ``.sciworld.json``. Raises :class:`GitHubError` on failure.
    """
    source_dir = Path(source_dir)
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    sweep_stale_temp(source_dir.parent)
    tmp = source_dir.parent / f".{source_dir.name}.tmp-{os.getpid()}"
    old = source_dir.parent / f".{source_dir.name}.old-{os.getpid()}"
    remove_tree(tmp)
    try:
        clone_repo(clone_url, tmp, token=github_token())
        record = {
            "arxiv_id": arxiv_id,
            "clone_url": clone_url,
            "commit": _head_commit(tmp),
            "cloned_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        remove_tree(tmp / ".git")
        (tmp / SOURCE_RECORD).write_text(json.dumps(record, indent=2) + "\n")
        if os.path.lexists(str(source_dir)):
            if not replace:
                return  # another process finished first
            remove_tree(old)
            os.replace(source_dir, old)
            try:
                os.replace(tmp, source_dir)
            except OSError:
                os.replace(old, source_dir)  # put the previous one back
                raise
        else:
            os.replace(tmp, source_dir)
    finally:
        for leftover in (tmp, old):
            try:
                remove_tree(leftover)
            except OSError:
                pass
