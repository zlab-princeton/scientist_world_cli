"""``sw select``: browse the environments and download one."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from sciworld import style as S
from sciworld.settings import ConfigError, load_config
from sciworld.github import DEFAULT_ORG, GitHubError, github_token
from sciworld.papers import (
    check_source, clone_environment, fetch_papers, lookup_paper, normalize_arxiv_id,
)
from sciworld.widgets import dice_select, paginated_list, tabbed_search


def _fail(message: str, args=None) -> int:
    S.bar()
    S.step_done(S.error(message))
    print()
    if args is not None:
        args._sw_error = S.plain(message)
    return 1


def _pick(papers: list) -> Optional[dict]:
    while True:
        S.bar()
        choice = S.select("How would you like to find a paper?", [
            "Random (dice roll)",
            "Search (by ID, time, or topic)",
            "List all (paginated)",
            "Exit",
        ])
        S.bar()
        if choice == 3:
            return None
        selected = (dice_select, tabbed_search, paginated_list)[choice](papers)
        if selected is not None:
            return selected


def cmd_select(args) -> int:
    """Choose a paper and download its environment to
    ``<data_dir>/envs/<id>/source``, or to ``--path``."""
    def fail(message: str) -> int:
        return _fail(message, args)

    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"  {S.error(str(exc))}")
        args._sw_error = str(exc)
        return 1
    org = getattr(args, "org", None) or cfg.get("org") or DEFAULT_ORG
    paper_id = getattr(args, "paper_id", None)
    dest_path = getattr(args, "dest_path", None)
    refresh = bool(getattr(args, "refresh", False))
    overwrite = bool(getattr(args, "overwrite", False))
    data_dir = cfg.get("data_dir") or ""

    S.intro("Scientist World", subtitle="Select a paper environment")
    if not dest_path and not data_dir:
        return fail(
            f"Data directory not set. Run {S.BOLD}sw init{S.RESET} or pass {S.BOLD}--path{S.RESET}."
        )

    if paper_id:
        paper_id = normalize_arxiv_id(paper_id)
        try:
            selected = lookup_paper(paper_id, org)
        except GitHubError as exc:
            return fail(str(exc))
        if selected is None:
            hint = "" if github_token() else " Without a GitHub token only public environments are visible."
            return fail(f"No environment found for {S.BOLD}{paper_id}{S.RESET} in {org}.{hint}")
    else:
        if not S.is_interactive():
            return fail("No paper given. Pass an arXiv id, or use `sw list` to see what is available.")
        papers = fetch_papers(org, refresh=refresh)
        if papers is None:
            print()
            args._sw_error = f"could not list the environments of {org}"
            return 1
        if not papers:
            return fail("No environments found.")
        selected = _pick(papers)
        if selected is None:
            S.step_done(f"{S.DIM}No paper selected.{S.RESET}")
            print()
            return 0

    arxiv_id = selected["arxiv_id"]
    S.bar()
    S.step_done(f"Selected {S.BOLD}{arxiv_id}{S.RESET}  {S.DIM}{selected.get('title', '')}{S.RESET}")

    target = (
        Path(dest_path).expanduser().resolve() if dest_path
        else Path(data_dir) / "envs" / arxiv_id / "source"
    )
    args._sw_result = target
    next_step = (
        f"sw research {arxiv_id}" if not dest_path
        else f"sw research {arxiv_id} --path {target.parent / 'RUN_NAME'}"
    ) if target.name == "source" else ""

    replace = False
    if os.path.lexists(str(target)):
        empty = target.is_dir() and not target.is_symlink() and not any(target.iterdir())
        if empty:
            target.rmdir()
        else:
            # Only a downloaded environment of this paper may be reused or
            # replaced. Anything else at that path is not ours to touch.
            problem = check_source(target, arxiv_id)
            if problem:
                return fail(f"{problem} Choose another {S.BOLD}--path{S.RESET}.")
            if not overwrite and S.is_interactive():
                S.bar()
                choice = S.select(f"{target} already holds this environment", ["Keep it", "Download again"])
                overwrite = choice == 1
            if not overwrite:
                S.bar()
                S.step_done(S.success(f"Already downloaded at {S.BOLD}{target}{S.RESET}"))
                S.step_end(f"{S.DIM}Next: {S.RESET}{S.BOLD}{next_step}{S.RESET}" if next_step else "")
                print()
                return 0
            replace = True

    clone_url = selected.get("clone_url") or ""
    if not clone_url:
        return fail("No clone URL available for this paper.")
    S.bar()
    S.step_start("Downloading the environment")
    try:
        # The download goes to a temporary directory first: if it fails, an
        # environment that is already there stays as it was.
        clone_environment(clone_url, target, arxiv_id=arxiv_id, replace=replace)
    except (GitHubError, OSError) as exc:
        return fail(str(exc))
    S.step_done(S.success(f"Ready at {S.BOLD}{target}{S.RESET}"))
    S.bar()
    if not dest_path:
        S.note([
            f"Details: {S.RESET}{S.BOLD}sw info {arxiv_id}{S.RESET}",
            f"Start:   {S.RESET}{S.BOLD}{next_step}{S.RESET}",
        ])
    elif next_step:
        S.note([f"Start:   {S.RESET}{S.BOLD}{next_step}{S.RESET}"])
    else:
        S.note([
            "To build workspaces from this download, keep it in a directory named",
            f"{S.RESET}source{S.DIM} and pass {S.RESET}sw research {arxiv_id} --path <that directory's parent>/RUN_NAME",
        ])
    S.step_end("")
    print()
    return 0
