"""``sw list``: print the available environments."""

from __future__ import annotations

import json

from sciworld import style as S
from sciworld.settings import ConfigError, load_config
from sciworld.github import DEFAULT_ORG
from sciworld.milestones import MILESTONE_ORDER, meets_minimum, milestone_rank
from sciworld.papers import fetch_papers


def filter_papers(papers: list, min_milestone: str = "", search: str = "") -> list:
    """Papers at or above *min_milestone* whose id, title or topics contain
    every word of *search*."""
    words = [w for w in (search or "").lower().split() if w]
    out = []
    for paper in papers:
        if min_milestone and not meets_minimum(paper.get("milestone", "none"), min_milestone):
            continue
        haystack = " ".join(
            [paper.get("arxiv_id", ""), paper.get("title", "")] + list(paper.get("topics", []))
        ).lower()
        if all(word in haystack for word in words):
            out.append(paper)
    return out


def cmd_list(args) -> int:
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"  {S.error(str(exc))}")
        return 1
    org = getattr(args, "org", None) or cfg.get("org") or DEFAULT_ORG
    as_json = bool(getattr(args, "json", False))
    limit = getattr(args, "limit", None)

    if not as_json:
        S.intro("Scientist World", subtitle=f"Environments of {org}")
    papers = fetch_papers(org, refresh=bool(getattr(args, "refresh", False)), quiet=as_json)
    if papers is None:
        if as_json:
            print(json.dumps({"error": f"could not list the environments of {org}"}))
        else:
            print()
        return 1

    selected = filter_papers(
        papers, getattr(args, "min_milestone", "") or "", getattr(args, "search", "") or "",
    )
    selected.sort(key=lambda p: (-milestone_rank(p.get("milestone", "none")), p.get("arxiv_id", "")))
    shown = selected[:limit] if limit else selected

    if as_json:
        print(json.dumps(shown, indent=2))
        return 0

    if not shown:
        S.bar()
        S.step_end(f"{S.DIM}No environment matches.{S.RESET}")
        print()
        return 0
    print()
    S.table_header(("ARXIV ID", 14), ("MILESTONE", 22), ("TITLE", 60))
    for paper in shown:
        title = paper.get("title", "")
        if len(title) > 58:
            title = title[:57] + "…"
        S.table_row(
            [f"{S.BRIGHT_WHITE}{paper['arxiv_id']}{S.RESET}",
             S.colored_milestone(paper.get("milestone", "none")),
             f"{S.DIM}{title}{S.RESET}"],
            [14, 22, 60],
        )
    print()
    more = f" (showing {len(shown)})" if len(shown) < len(selected) else ""
    print(f"  {S.DIM}{len(selected)} environments{more}. "
          f"Start one with: {S.RESET}sw research <arxiv_id>")
    print()
    return 0


MILESTONE_CHOICES = list(MILESTONE_ORDER)
