"""Interactive selection widgets for paper browsing."""

from __future__ import annotations

import colorsys
import random
import sys
import textwrap
from datetime import datetime, timezone

from sciworld.style import (
    _IS_TTY,
    _G_RADIO_ON, _G_RADIO_OFF,
    _P_RADIO_ON, _P_RADIO_OFF,
    _read_key, _read_key_timeout, clear_lines, _visible_len, _term_cols,
    _gradient_line,
    RESET, BOLD, DIM, ITALIC,
    HIDE_CURSOR, SHOW_CURSOR,
    BRIGHT_CYAN, BRIGHT_WHITE,
    BOX_TL, BOX_TR, BOX_BL, BOX_BR, BOX_H, BOX_V,
    _C1, _C2,
)


# ── Helpers ─────────────────────────────────────────────────


def _truncate(text: str, width: int) -> str:
    """Truncate text to width, appending ellipsis if needed."""
    if len(text) <= width:
        return text
    return text[: width - 1] + "\u2026"


_STRIP_PREFIXES = ("area-", "domain-", "category-", "method-",
                    "secondary-area-", "secondary-category-",
                    "secondary-domain-", "secondary-method-")

_SKIP_TOPICS = {"done", "paper-type-metric-method", "paper-type-theory",
                "paper-type-empirical"}


def _paper_topics_str(paper: dict) -> str:
    """Format topics as comma-separated string, with prefixes stripped."""
    topics = paper.get("topics", [])
    display = []
    seen = set()
    for t in topics:
        if t in _SKIP_TOPICS or t.startswith("iteration-") or t.startswith("milestone-") or t.startswith("selection-"):
            continue
        label = t
        for prefix in _STRIP_PREFIXES:
            if t.startswith(prefix):
                label = t[len(prefix):]
                break
        label = label.replace("-", " ")
        if label not in seen:
            seen.add(label)
            display.append(label)
    return ", ".join(display[:5])


def _random_gradient() -> tuple:
    """Generate a random complementary color pair for gradients."""
    hue = random.random()
    r, g, b = colorsys.hls_to_rgb(hue, 0.65, 0.8)
    c1 = (int(r * 255), int(g * 255), int(b * 255))
    hue2 = (hue + 0.3 + random.random() * 0.2) % 1.0
    r, g, b = colorsys.hls_to_rgb(hue2, 0.65, 0.8)
    c2 = (int(r * 255), int(g * 255), int(b * 255))
    return c1, c2


# ── Widget 1: dice_select ───────────────────────────────────


# TLDR cache: repo_name -> tldr string
_tldr_cache: dict[str, str] = {}


def _get_tldr(paper: dict) -> str:
    """Get TLDR for a paper, fetching from GitHub if needed."""
    repo_name = paper.get("repo_name", "")
    if not repo_name:
        return ""
    if repo_name in _tldr_cache:
        return _tldr_cache[repo_name]
    try:
        from sciworld.github import fetch_repo_tldr
        tldr = fetch_repo_tldr(repo_name)
    except Exception:
        tldr = ""
    _tldr_cache[repo_name] = tldr
    return tldr


def _card_width() -> int:
    """Dynamic card inner width: uses most of the terminal."""
    return min(_term_cols() - 4, 78)


def _card_line(left: str, content: str, right: str, w: int) -> str:
    """Build a card line: left border + content (padded to w) + right border."""
    vlen = _visible_len(content)
    pad = max(w - vlen, 0)
    return f"  {left}{content}{' ' * pad}{right}"


def _draw_paper_card(paper: dict, colors: tuple | None = None, tldr: str = "") -> list[str]:
    """Build the lines for a paper card (no printing)."""
    c1, c2 = colors or (_C1, _C2)
    w = _card_width()
    lines = []

    lv = _gradient_line(BOX_V, c1, c1)
    rv = _gradient_line(BOX_V, c2, c2)

    # Top border
    lines.append(f"  {_gradient_line(f'{BOX_TL}{BOX_H * w}{BOX_TR}', c1, c2)}")

    # Empty line
    lines.append(f"  {lv}{' ' * w}{rv}")

    # Arxiv ID
    aid = paper.get("arxiv_id", "")
    inner = f"  {BRIGHT_WHITE}{aid}{RESET}"
    lines.append(_card_line(lv, inner, rv, w))

    # Title: wrapped to fit card
    title = paper.get("title", "")
    title_w = w - 4  # 2 spaces padding each side
    for tl in textwrap.wrap(title, width=title_w) or [title]:
        inner = f"  {BOLD}{tl}{RESET}"
        lines.append(_card_line(lv, inner, rv, w))

    # Empty line
    lines.append(f"  {lv}{' ' * w}{rv}")

    # TLDR (wrapped)
    if tldr:
        tldr_w = w - 4
        for tl in textwrap.wrap(tldr, width=tldr_w):
            inner = f"  {DIM}{ITALIC}{tl}{RESET}"
            lines.append(_card_line(lv, inner, rv, w))
        lines.append(f"  {lv}{' ' * w}{rv}")

    # Topics
    topics_str = _paper_topics_str(paper)
    if topics_str:
        topics_w = w - 4
        for tl in textwrap.wrap(topics_str, width=topics_w):
            inner = f"  {DIM}{tl}{RESET}"
            lines.append(_card_line(lv, inner, rv, w))

    # URL
    url = f"https://arxiv.org/abs/{aid}"
    inner = f"  {DIM}{url}{RESET}"
    lines.append(_card_line(lv, inner, rv, w))

    # Empty line
    lines.append(f"  {lv}{' ' * w}{rv}")

    # Bottom border
    lines.append(f"  {_gradient_line(f'{BOX_BL}{BOX_H * w}{BOX_BR}', c1, c2)}")

    return lines


_DICE_FRAMES = ["⚀", "⚁", "⚂", "⚃", "⚄", "⚅"]


def dice_select(papers: list) -> dict | None:
    """Show a random paper with Accept / Roll again / Back options."""
    if not papers:
        return None

    if not _IS_TTY:
        return papers[0] if papers else None

    _OPT_ACCEPT = 0
    _OPT_ROLL = 1
    opt_labels = ["Accept", "Roll again", "Back"]
    selected = 0
    dice_frame = 0
    seen: set[str] = set()  # track shown paper IDs to avoid repeats
    colors = _random_gradient()
    current_paper = random.choice(papers)
    seen.add(current_paper.get("arxiv_id", ""))
    current_tldr = _get_tldr(current_paper)

    prev_lines = [0]  # mutable so nested fn can update

    def _draw_frame(first: bool = False) -> int:
        """Draw card + menu. Returns total lines drawn."""
        card_lines = _draw_paper_card(current_paper, colors=colors, tldr=current_tldr)
        menu_lines = len(opt_labels) + 1  # options + hint
        total = len(card_lines) + 1 + menu_lines  # card + blank + menu

        if not first:
            clear_lines(prev_lines[0])

        for line in card_lines:
            print(line)
        print()

        _draw_menu()

        prev_lines[0] = total
        return total

    def _draw_menu() -> None:
        """Draw just the option lines + hint."""
        for i, label in enumerate(opt_labels):
            if i == _OPT_ROLL:
                # Animate dice only while hovering on this option
                if selected == _OPT_ROLL:
                    face = _DICE_FRAMES[dice_frame % len(_DICE_FRAMES)]
                else:
                    face = "\U0001f3b2"
                display = f"{face} {label}"
            else:
                display = label

            if i == selected:
                print(f"  {BRIGHT_CYAN}{_G_RADIO_ON}{RESET}{_P_RADIO_ON}{BOLD}{display}{RESET}")
            else:
                print(f"  {DIM}{_G_RADIO_OFF}{_P_RADIO_OFF}{display}{RESET}")

        print(f"  {DIM}\u2191\u2193 navigate   enter select   esc back{RESET}")

    def _redraw_menu() -> None:
        """Redraw only the menu lines (options + hint) in place."""
        menu_lines = len(opt_labels) + 1
        clear_lines(menu_lines)
        _draw_menu()

    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.flush()

    try:
        total_lines = _draw_frame(first=True)

        while True:
            # When hovering on Roll, animate the dice; otherwise block
            if selected == _OPT_ROLL:
                key = _read_key_timeout(0.15)
                if key is None:
                    dice_frame += 1
                    _redraw_menu()
                    continue
            else:
                key = _read_key()

            if key == "up":
                selected = (selected - 1) % len(opt_labels)
                _redraw_menu()
            elif key == "down":
                selected = (selected + 1) % len(opt_labels)
                _redraw_menu()
            elif key == "enter":
                if selected == _OPT_ACCEPT:
                    clear_lines(total_lines)
                    return current_paper
                elif selected == _OPT_ROLL:
                    unseen = [p for p in papers if p.get("arxiv_id", "") not in seen]
                    if not unseen:
                        seen.clear()  # all shown, reset
                        unseen = papers
                    current_paper = random.choice(unseen)
                    seen.add(current_paper.get("arxiv_id", ""))
                    colors = _random_gradient()
                    current_tldr = _get_tldr(current_paper)
                    selected = 0
                    total_lines = _draw_frame()
                else:
                    clear_lines(total_lines)
                    return None
            elif key in ("q", "escape"):
                clear_lines(total_lines)
                return None
            elif key == "interrupt":
                clear_lines(total_lines)
                raise KeyboardInterrupt
    finally:
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()


# ── Widget 2: paginated_list ────────────────────────────────


_SORT_MODES = ["id", "date"]


def _sort_papers(papers: list, mode: str) -> list:
    """Sort papers by the given mode."""
    if mode == "id":
        return sorted(papers, key=lambda p: p.get("arxiv_id", ""))
    elif mode == "date":
        return sorted(papers, key=lambda p: p.get("pushed_at", "") or p.get("created_at", ""), reverse=True)
    return papers


def _list_width() -> int:
    """Usable width for paper list rows."""
    return min(_term_cols() - 4, 90)


def _paper_row(paper: dict, selected: bool, w: int) -> str:
    """Render one paper row: marker + id + title."""
    marker = f"{BRIGHT_CYAN}›{RESET}" if selected else " "
    aid = paper.get("arxiv_id", "")
    title_w = w - 16  # marker(2) + id(12) + gaps(2)
    title = _truncate(paper.get("title", ""), title_w)
    if selected:
        return f"  {marker} {BRIGHT_WHITE}{aid:<12}{RESET}  {title}"
    else:
        return f"  {marker} {DIM}{aid:<12}{RESET}  {DIM}{title}{RESET}"


def paginated_list(papers: list, page_size: int = 20) -> dict | None:
    """Paginated table of papers with sort cycling."""
    if not papers:
        return None

    if not _IS_TTY:
        return papers[0] if papers else None

    w = _list_width()
    sort_mode = 0
    sorted_papers = _sort_papers(papers, _SORT_MODES[sort_mode])
    cursor = 0
    page = 0
    total_pages = max(1, (len(sorted_papers) + page_size - 1) // page_size)

    def _get_page_items() -> list:
        start = page * page_size
        return sorted_papers[start : start + page_size]

    def _draw_frame(first: bool = False) -> int:
        nonlocal total_pages
        total_pages = max(1, (len(sorted_papers) + page_size - 1) // page_size)
        items = _get_page_items()
        total = 2 + page_size + 2  # header + sep + rows (padded) + sep + hint

        if not first:
            clear_lines(total)

        sort_label = _SORT_MODES[sort_mode]
        count = len(sorted_papers)
        print(f"  {DIM}{count} papers{RESET}   "
              f"{DIM}page {BRIGHT_WHITE}{page + 1}{RESET}{DIM}/{total_pages}{RESET}   "
              f"{DIM}sort: {BRIGHT_WHITE}{sort_label}{RESET}")
        print(f"  {DIM}{'─' * w}{RESET}")

        for i, paper in enumerate(items):
            print(_paper_row(paper, i == cursor, w))
        for _ in range(page_size - len(items)):
            print()

        print(f"  {DIM}{'─' * w}{RESET}")
        print(f"  {DIM}\u2191\u2193 navigate   n/p page   s sort   enter select   esc back{RESET}")
        return total

    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.flush()

    try:
        total_lines = _draw_frame(first=True)

        while True:
            key = _read_key()
            items = _get_page_items()

            if key == "up":
                if cursor > 0:
                    cursor -= 1
                elif page > 0:
                    page -= 1
                    cursor = min(page_size - 1, len(_get_page_items()) - 1)
                _draw_frame()
            elif key == "down":
                if cursor < len(items) - 1:
                    cursor += 1
                elif page < total_pages - 1:
                    page += 1
                    cursor = 0
                _draw_frame()
            elif key in ("n", "page-down"):
                if page < total_pages - 1:
                    page += 1
                    cursor = 0
                _draw_frame()
            elif key in ("p", "page-up"):
                if page > 0:
                    page -= 1
                    cursor = 0
                _draw_frame()
            elif key == "s":
                sort_mode = (sort_mode + 1) % len(_SORT_MODES)
                sorted_papers = _sort_papers(papers, _SORT_MODES[sort_mode])
                page = 0
                cursor = 0
                _draw_frame()
            elif key == "enter":
                if items:
                    clear_lines(total_lines)
                    return items[cursor]
            elif key in ("q", "escape"):
                clear_lines(total_lines)
                return None
            elif key == "interrupt":
                clear_lines(total_lines)
                raise KeyboardInterrupt
    finally:
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()


# ── Widget 3: tabbed_search ─────────────────────────────────


_TAB_NAMES = ["By ID", "By Time", "By Topic"]


def _paper_date(paper: dict):
    """When the environment was last updated, or ``None`` if unknown."""
    text = paper.get("pushed_at", "") or paper.get("created_at", "")
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "").split("+")[0])
    except (ValueError, TypeError):
        return None


def _time_buckets(papers: list) -> list:
    """Buckets of the "By Time" tab as ``(label, kind, value)``: two recent
    windows, then one bucket per year that has papers, newest first, and one
    for papers without a date. Every paper falls in at least one bucket."""
    dates = [_paper_date(p) for p in papers]
    buckets = [("Last 30 days", "days", 30), ("Last 90 days", "days", 90)]
    buckets += [(str(year), "year", year) for year in sorted({d.year for d in dates if d}, reverse=True)]
    if any(d is None for d in dates):
        buckets.append(("No date", "none", None))
    return buckets


def _papers_in_time_bucket(papers: list, kind: str, value) -> list:
    """Papers that belong to one bucket of :func:`_time_buckets`."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    results = []
    for paper in papers:
        date = _paper_date(paper)
        if kind == "none":
            keep = date is None
        elif date is None:
            keep = False
        elif kind == "days":
            keep = (now - date).days <= value
        else:
            keep = date.year == value
        if keep:
            results.append(paper)
    return results


def _window(count: int, cursor: int, height: int) -> int:
    """Index of the first visible row of a list of *count* rows shown
    *height* at a time, such that row *cursor* is visible."""
    if count <= height:
        return 0
    return (cursor // height) * height


def _extract_topic_groups(papers: list) -> list:
    """Get unique display-friendly topic labels with their raw keys."""
    seen: dict[str, str] = {}  # display_label -> raw_topic
    for p in papers:
        for t in p.get("topics", []):
            for prefix in _STRIP_PREFIXES:
                if t.startswith(prefix):
                    label = t[len(prefix):].replace("-", " ")
                    if label not in seen:
                        seen[label] = t
                    break
    return sorted(seen.items())  # list of (display_label, raw_topic)


def _papers_with_topic(papers: list, topic: str) -> list:
    """Filter papers that have the given topic."""
    return [p for p in papers if topic in p.get("topics", [])]


def _tab_bar(active: int) -> str:
    """Render a clean tab bar."""
    parts = []
    for i, name in enumerate(_TAB_NAMES):
        if i == active:
            parts.append(f"{BRIGHT_CYAN}{BOLD}{name}{RESET}")
        else:
            parts.append(f"{DIM}{name}{RESET}")
    return "  " + "    ".join(parts)


def tabbed_search(papers: list) -> dict | None:
    """Three-tab search widget: By ID, By Time, By Topic."""
    if not papers:
        return None

    if not _IS_TTY:
        return papers[0] if papers else None

    w = _list_width()
    active_tab = 0
    query = ""
    cursor = 0           # position in the By ID results
    sub_list = None      # type: list | None   papers of the opened bucket or topic
    sub_cursor = 0
    bucket_cursor = 0    # position in the bucket or topic list

    topic_groups = _extract_topic_groups(papers)
    time_buckets = _time_buckets(papers)
    viewport_h = 15

    def _get_id_results() -> list:
        if not query:
            return papers
        q = query.replace("_", ".").lower()
        return [p for p in papers
                if q in p.get("arxiv_id", "").lower()
                or q in p.get("title", "").lower()]

    def _rows(items: list, position: int, render) -> list:
        """*viewport_h* lines: the page of *items* that holds *position*."""
        first = _window(len(items), position, viewport_h)
        lines = [render(item, first + i == position) for i, item in enumerate(items[first:first + viewport_h])]
        lines += [""] * (viewport_h - len(lines))
        return lines

    def _label_row(label: str, selected: bool) -> str:
        if selected:
            return f"  {BRIGHT_CYAN}›{RESET} {BOLD}{label}{RESET}"
        return f"    {DIM}{label}{RESET}"

    def _count(shown: int, total: int, what: str) -> str:
        more = f"   {DIM}{position_hint(shown, total)}{RESET}" if total > viewport_h else ""
        return f"{DIM}{total} {what}{RESET}{more}"

    def position_hint(position: int, total: int) -> str:
        first = _window(total, position, viewport_h)
        return f"{first + 1}-{min(first + viewport_h, total)} shown"

    def _draw_frame(first: bool = False) -> int:
        lines_out: list[str] = []
        lines_out.append(_tab_bar(active_tab))
        lines_out.append(f"  {DIM}{'─' * w}{RESET}")
        lines_out.append("")

        if active_tab == 0:
            results = _get_id_results()
            lines_out.append(f"  {DIM}Search:{RESET} {BRIGHT_WHITE}{query}{RESET}{BRIGHT_CYAN}\u2588{RESET}"
                             f"   {_count(cursor, len(results), 'papers')}")
            lines_out.append("")
            lines_out += _rows(results, cursor, lambda p, sel: _paper_row(p, sel, w))
        else:
            groups = time_buckets if active_tab == 1 else topic_groups
            if sub_list is None:
                lines_out.append(f"  {_count(bucket_cursor, len(groups), 'groups')}")
                lines_out.append("")
                lines_out += _rows(groups, bucket_cursor, lambda g, sel: _label_row(g[0], sel))
            else:
                label = groups[bucket_cursor][0] if bucket_cursor < len(groups) else ""
                lines_out.append(f"  {DIM}{label}{RESET}   {_count(sub_cursor, len(sub_list), 'papers')}")
                lines_out.append("")
                lines_out += _rows(sub_list, sub_cursor, lambda p, sel: _paper_row(p, sel, w))

        lines_out.append("")
        lines_out.append(f"  {DIM}{'─' * w}{RESET}")
        lines_out.append(f"  {DIM}\u2191\u2193 navigate   tab switch   enter select   esc back{RESET}")

        total = len(lines_out)
        if not first:
            clear_lines(total)
        for line in lines_out:
            print(line)
        return total

    def _step(position: int, count: int, key: str) -> int:
        if key == "up":
            return max(position - 1, 0)
        return min(position + 1, max(count - 1, 0))

    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.flush()

    try:
        total_lines = _draw_frame(first=True)

        while True:
            key = _read_key()

            if key in ("tab", "shift-tab"):
                step = 1 if key == "tab" else -1
                active_tab = (active_tab + step) % len(_TAB_NAMES)
                cursor = 0
                sub_list = None
                sub_cursor = 0
                bucket_cursor = 0
                query = ""
                total_lines = _draw_frame()

            elif key == "escape":
                if sub_list is not None:
                    sub_list = None
                    sub_cursor = 0
                    total_lines = _draw_frame()
                else:
                    clear_lines(total_lines)
                    return None

            elif key == "interrupt":
                clear_lines(total_lines)
                raise KeyboardInterrupt

            elif active_tab == 0:
                results = _get_id_results()
                if key == "backspace":
                    query = query[:-1]
                    cursor = 0
                    total_lines = _draw_frame()
                elif key in ("up", "down"):
                    cursor = _step(cursor, len(results), key)
                    total_lines = _draw_frame()
                elif key == "enter":
                    if cursor < len(results):
                        clear_lines(total_lines)
                        return results[cursor]
                elif len(key) == 1 and key.isprintable():
                    query += key
                    cursor = 0
                    total_lines = _draw_frame()

            else:
                groups = time_buckets if active_tab == 1 else topic_groups
                if sub_list is None:
                    if key in ("up", "down"):
                        bucket_cursor = _step(bucket_cursor, len(groups), key)
                        total_lines = _draw_frame()
                    elif key == "enter" and bucket_cursor < len(groups):
                        if active_tab == 1:
                            _label, kind, value = groups[bucket_cursor]
                            sub_list = _papers_in_time_bucket(papers, kind, value)
                        else:
                            sub_list = _papers_with_topic(papers, groups[bucket_cursor][1])
                        sub_cursor = 0
                        total_lines = _draw_frame()
                else:
                    if key in ("up", "down"):
                        sub_cursor = _step(sub_cursor, len(sub_list), key)
                        total_lines = _draw_frame()
                    elif key == "enter" and sub_cursor < len(sub_list):
                        clear_lines(total_lines)
                        return sub_list[sub_cursor]
    finally:
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()
