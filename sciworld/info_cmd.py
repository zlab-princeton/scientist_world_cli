"""``sw info``: show what a downloaded environment is about."""

from __future__ import annotations

import re
from pathlib import Path

from sciworld import style as S
from sciworld.settings import ConfigError, load_config
from sciworld.papers import normalize_arxiv_id


# ── Markdown parsing helpers ─────────────────────────────────


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def _strip_comments(text: str) -> str:
    """Remove HTML comments used as placeholder instructions in templates."""
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def _first_paragraph(text: str) -> str:
    """Return the first non-heading, non-empty paragraph."""
    text = _strip_comments(text)
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#"):
            continue
        return " ".join(block.split())
    return ""


def _extract_section(text: str, heading: str) -> str:
    """Return the content under a markdown heading (## or ###)."""
    text = _strip_comments(text)
    # Match heading at any level, capture until next heading or EOF
    pattern = rf"^#{{1,3}}\s*{re.escape(heading)}\s*\n(.*?)(?=^#{{1,3}}\s|\Z)"
    m = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def _parse_overview(text: str) -> dict:
    """Parse overview.md which has key:value bullets."""
    text = _strip_comments(text)
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"\s*[-*]\s*\*\*([^:*]+):\*\*\s*(.+)$", line)
        if m:
            key = m.group(1).strip().lower().replace(" ", "_")
            out[key] = m.group(2).strip()
    # Short summary section
    summary = _extract_section(text, "Short Summary")
    if summary:
        out["short_summary"] = " ".join(summary.split())
    return out


def _parse_experiments_md(text: str) -> list[tuple[str, str]]:
    """Extract (name, description) pairs from EXPERIMENTS.md."""
    text = _strip_comments(text)
    experiments = []
    # Each experiment is under `## name` or `## \`name\``
    for m in re.finditer(
        r"^##\s*`?([^`\n]+?)`?\s*\n(.*?)(?=^##\s|\Z)",
        text, flags=re.MULTILINE | re.DOTALL,
    ):
        name = m.group(1).strip()
        body = m.group(2).strip()
        # First paragraph is the description
        first = body.split("\n\n", 1)[0].strip()
        first = " ".join(first.split())
        experiments.append((name, first))
    return experiments


# ── Rendering ────────────────────────────────────────────────


def _section(title: str) -> None:
    S.bar()
    S.bar(f"{S.BOLD}{title}{S.RESET}")


def _multiline(text: str, style: str = "") -> None:
    """Print a block of text wrapped under the bar."""
    if not text:
        return
    import textwrap
    cols = S._term_cols()
    usable = max(cols - 6 - 4, 20)
    prefix = f"{S._M}{S.DIM}{S._G_BAR}{S.RESET}{S._P_BAR}"
    for para in re.split(r"\n\s*\n", text.strip()):
        para = " ".join(para.split())
        for line in textwrap.wrap(para, width=usable):
            print(f"{prefix}{style}{line}{S.RESET if style else ''}")
        print(f"{S._M}{S.DIM}{S._G_BAR}{S.RESET}")


# ── Main command ─────────────────────────────────────────────


def cmd_info(args) -> int:
    """Show paper information."""
    arxiv_id = getattr(args, "arxiv_id", None)
    full = getattr(args, "full", False)

    def _fail(message: str) -> int:
        S.intro("Scientist World")
        S.bar()
        S.step_done(S.error(message))
        print()
        args._sw_error = S.plain(message)
        return 1

    if not arxiv_id:
        return _fail(f"Missing arXiv id. Usage: {S.BOLD}sw info <arxiv_id>{S.RESET}")
    arxiv_id = normalize_arxiv_id(arxiv_id)

    try:
        cfg = load_config()
    except ConfigError as exc:
        return _fail(str(exc))
    data_dir = cfg.get("data_dir") or ""
    if not data_dir:
        return _fail(f"Data directory not set. Run {S.BOLD}sw init{S.RESET}.")

    source = Path(data_dir) / "envs" / arxiv_id / "source"
    if not source.is_dir():
        return _fail(
            f"{S.BOLD}{arxiv_id}{S.RESET} is not downloaded yet. "
            f"Run {S.BOLD}sw select {arxiv_id}{S.RESET} first."
        )

    # Load files
    overview_md = _read(source / "briefing" / "overview.md")
    problem_md = _read(source / "briefing" / "problem.md")
    method_md = _read(source / "briefing" / "method.md")
    evaluation_md = _read(source / "briefing" / "evaluation.md")
    experiments_md = _read(source / "scoring" / "EXPERIMENTS.md")
    targets_md = _read(source / "scoring" / "TARGETS.md")
    constraints_md = _read(source / "scoring" / "CONSTRAINTS.md")
    direction_md = _read(source / "scoring" / "DIRECTION.md")

    overview = _parse_overview(overview_md)
    title = overview.get("title", "")
    subtitle = title or f"Paper {arxiv_id}"
    S.intro("Scientist World", subtitle=subtitle)

    # ── Metadata ──
    _section("Metadata")
    S.bar(f"  {S.DIM}arXiv    {S.RESET}{arxiv_id}")
    if overview.get("domain"):
        S.bar(f"  {S.DIM}domain   {S.RESET}{overview['domain']}")
    S.bar(f"  {S.DIM}url      {S.RESET}{S.DIM}https://arxiv.org/abs/{arxiv_id}{S.RESET}")
    S.bar(f"  {S.DIM}source   {S.RESET}{S.DIM}{source}{S.RESET}")

    from sciworld.evaluate_cmd import list_runs
    runs = list_runs(source.parent)
    if runs:
        S.bar(f"  {S.DIM}runs     {S.RESET}{', '.join(runs)}")

    # ── TL;DR ──
    tldr = overview.get("tl;dr") or overview.get("tldr", "")
    if tldr:
        _section("TL;DR")
        _multiline(tldr)

    # ── Short Summary ──
    short_summary = overview.get("short_summary", "")
    if short_summary:
        _section("Summary")
        _multiline(short_summary)

    # ── Problem ──
    problem = _first_paragraph(problem_md) if problem_md else ""
    if problem:
        _section("Problem")
        _multiline(problem)

    # ── Experiments ──
    experiments = _parse_experiments_md(experiments_md) if experiments_md else []
    if experiments:
        _section(f"Experiments ({len(experiments)})")
        for name, desc in experiments:
            S.bar(f"  {S.BRIGHT_CYAN}●{S.RESET} {S.BOLD}{name}{S.RESET}")
            if desc:
                import textwrap
                usable = max(S._term_cols() - 12 - 4, 20)
                for line in textwrap.wrap(desc, width=usable):
                    S.bar(f"       {S.DIM}{line}{S.RESET}")

    # ── Full mode: include method, targets, constraints, direction ──
    if full:
        if method_md:
            _section("Method")
            approach = _extract_section(method_md, "Approach")
            if approach:
                _multiline(approach, style=S.DIM)
            else:
                _multiline(_first_paragraph(method_md), style=S.DIM)

        if targets_md:
            _section("Targets")
            primary = _extract_section(targets_md, "Primary Targets")
            _multiline(primary or _first_paragraph(targets_md), style=S.DIM)

        if constraints_md:
            _section("Constraints")
            _multiline(_first_paragraph(constraints_md), style=S.DIM)

        if direction_md:
            _section("Direction")
            _multiline(_first_paragraph(direction_md), style=S.DIM)

        if evaluation_md:
            _section("Evaluation")
            metrics = _extract_section(evaluation_md, "Metrics")
            _multiline(metrics or _first_paragraph(evaluation_md), style=S.DIM)
    else:
        S.bar()
        S.step_end(f"{S.DIM}Pass {S.RESET}{S.BOLD}--full{S.RESET}{S.DIM} for method, targets, constraints, evaluation{S.RESET}")

    print()
    return 0
