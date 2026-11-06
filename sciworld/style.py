"""Terminal styling: clack-inspired vertical flow with gradient header."""

from __future__ import annotations

import contextlib
import contextvars
import os
import re
import sys
# Disable colors if NO_COLOR is set or stdout is not a tty
_NO_COLOR = os.environ.get("NO_COLOR") or not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty()
_IS_TTY = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()

# Set while a function of the Python API runs: it must never ask a question,
# even when the calling script happens to run in a terminal.
_no_questions: contextvars.ContextVar = contextvars.ContextVar("sciworld_no_questions", default=False)


def _ansi(code: str) -> str:
    return "" if _NO_COLOR else code


def is_interactive() -> bool:
    """True when a person is at the terminal: stdin and stdout are both TTYs.

    Commands use this to decide between asking a question and failing with
    a message that names the flag to pass instead.
    """
    if _no_questions.get():
        return False
    return bool(
        _IS_TTY and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    )


@contextlib.contextmanager
def non_interactive():
    """Inside this block :func:`is_interactive` is false, whatever the terminal."""
    token = _no_questions.set(True)
    try:
        yield
    finally:
        _no_questions.reset(token)


def plain(text: str) -> str:
    """*text* without ANSI escape sequences."""
    return re.sub(r"\033\[[0-9;]*m", "", str(text))


def _rgb(r: int, g: int, b: int) -> str:
    """24-bit truecolor foreground."""
    return _ansi(f"\033[38;2;{r};{g};{b}m")


def _visible_len(text: str) -> int:
    """Length of text after stripping ANSI escape sequences."""
    return len(re.sub(r"\033\[[0-9;]*m", "", text))



# ── Core ANSI codes ──────────────────────────────────────

RESET = _ansi("\033[0m")
BOLD = _ansi("\033[1m")
DIM = _ansi("\033[2m")
ITALIC = _ansi("\033[3m")
UNDERLINE = _ansi("\033[4m")
HIDE_CURSOR = _ansi("\033[?25l")
SHOW_CURSOR = _ansi("\033[?25h")
CLEAR_LINE = _ansi("\033[2K")
MOVE_UP = _ansi("\033[1A")

RED = _ansi("\033[31m")
GREEN = _ansi("\033[32m")
YELLOW = _ansi("\033[33m")
BLUE = _ansi("\033[34m")
MAGENTA = _ansi("\033[35m")
CYAN = _ansi("\033[36m")

BRIGHT_RED = _ansi("\033[91m")
BRIGHT_GREEN = _ansi("\033[92m")
BRIGHT_YELLOW = _ansi("\033[93m")
BRIGHT_BLUE = _ansi("\033[94m")
BRIGHT_MAGENTA = _ansi("\033[95m")
BRIGHT_CYAN = _ansi("\033[96m")
BRIGHT_WHITE = _ansi("\033[97m")
BRIGHT_BLACK = _ansi("\033[90m")


# ── Glyphs + alignment ──────────────────────────────────
# Many terminals render geometric shapes (◆◇●○) as 2 columns
# but box-drawing (│└) as 1 column. We hardcode the widths and
# pad each glyph so text always starts at the same column.
#
#   ◆(2col) + 1sp  = 3 display cols → text starts at margin+3
#   │(1col) + 2sp  = 3 display cols → text starts at margin+3

_G_BAR = "│"
_G_END = "└"
_G_ACTIVE = "◆"
_G_DONE = "◇"
_G_RADIO_ON = "●"
_G_RADIO_OFF = "○"

# Narrow glyphs (box-drawing): 1 display column → 2 spaces after
_P_BAR = "  "
_P_END = "  "

# Wide glyphs (geometric shapes): 2 display columns → 1 space after
_P_ACTIVE = " "
_P_DONE = " "
_P_RADIO_ON = " "
_P_RADIO_OFF = " "

_M = "   "  # left margin

# ── Box-drawing (rounded corners) ────────────────────────

BOX_TL = "╭"
BOX_TR = "╮"
BOX_BL = "╰"
BOX_BR = "╯"
BOX_H = "─"
BOX_V = "│"


# ── Gradient rendering ───────────────────────────────────

_C1 = (80, 200, 255)   # cyan
_C2 = (180, 100, 255)  # purple


def _gradient_text(text: str, start: tuple, end: tuple) -> str:
    """Color each character along an RGB gradient."""
    if _NO_COLOR:
        return text
    n = max(len(text) - 1, 1)
    chars = []
    for i, ch in enumerate(text):
        if ch == " ":
            chars.append(ch)
            continue
        t = i / n
        r = int(start[0] + (end[0] - start[0]) * t)
        g = int(start[1] + (end[1] - start[1]) * t)
        b = int(start[2] + (end[2] - start[2]) * t)
        chars.append(f"{_rgb(r, g, b)}{ch}")
    chars.append(RESET)
    return "".join(chars)


def _gradient_line(text: str, start: tuple, end: tuple) -> str:
    """Apply gradient to every character including spaces."""
    if _NO_COLOR:
        return text
    n = max(len(text) - 1, 1)
    chars = []
    for i, ch in enumerate(text):
        t = i / n
        r = int(start[0] + (end[0] - start[0]) * t)
        g = int(start[1] + (end[1] - start[1]) * t)
        b = int(start[2] + (end[2] - start[2]) * t)
        chars.append(f"{_rgb(r, g, b)}{ch}")
    chars.append(RESET)
    return "".join(chars)


# ── Milestone styling ────────────────────────────────────

MILESTONE_COLORS = {
    "full": BRIGHT_GREEN,
    "near_complete": GREEN,
    "majority": CYAN,
    "secondary_claims": BRIGHT_BLUE,
    "core_claim_plus": BLUE,
    "core_claim": YELLOW,
    "method_runs": BRIGHT_YELLOW,
    "none": DIM,
    "not_possible": BRIGHT_RED,
}

MILESTONE_ICONS = {
    "full": "★",
    "near_complete": "◆",
    "majority": "●",
    "secondary_claims": "●",
    "core_claim_plus": "●",
    "core_claim": "○",
    "method_runs": "◌",
    "none": "·",
    "not_possible": "✖",
}


def colored_milestone(milestone: str) -> str:
    color = MILESTONE_COLORS.get(milestone, DIM)
    icon = MILESTONE_ICONS.get(milestone, " ")
    return f"{color}{icon} {milestone}{RESET}"


# ── Header / intro ───────────────────────────────────────

_HEADER_W = 66  # inner content width (between side borders)


def intro(title: str, subtitle: str = "", prefix: str = "") -> None:
    """Bordered gradient header with rounded corners."""
    print()

    # Top border: ╭──────────────╮
    top = f"{BOX_TL}{BOX_H * _HEADER_W}{BOX_TR}"
    print(f"  {_gradient_line(top, _C1, _C2)}")

    _hline("")

    # Title
    grad_title = _gradient_text(title, _C1, _C2)
    if prefix:
        _hline_raw(f"  {DIM}{prefix}{RESET} {BOLD}{grad_title}")
    else:
        _hline_raw(f"  {BOLD}{grad_title}")

    if subtitle:
        import textwrap as _tw
        _hline("")
        # Wrap each logical line to fit inside the box
        usable = _HEADER_W - 2  # 2-space left padding we add
        for raw_line in subtitle.split("\n"):
            raw_line = raw_line.strip()
            wrapped = _tw.wrap(raw_line, width=usable) if raw_line else [""]
            for wl in wrapped:
                _hline(f"  {DIM}{wl}{RESET}")

    _hline("")

    # Bottom border: ╰──────────────╯
    bottom = f"{BOX_BL}{BOX_H * _HEADER_W}{BOX_BR}"
    print(f"  {_gradient_line(bottom, _C1, _C2)}")
    print()


def _hline(content: str) -> None:
    """Print a header line with gradient side borders."""
    left = _gradient_text(BOX_V, _C1, _C1)
    right = _gradient_text(BOX_V, _C2, _C2)
    vlen = _visible_len(content)
    pad = max(_HEADER_W - vlen, 0)
    print(f"  {left}{content}{' ' * pad}{right}")


def _hline_raw(content: str) -> None:
    """Like _hline but content contains ANSI codes."""
    left = _gradient_text(BOX_V, _C1, _C1)
    right = _gradient_text(BOX_V, _C2, _C2)
    stripped = re.sub(r"\033\[[0-9;]*m", "", content)
    pad = max(_HEADER_W - len(stripped), 0)
    print(f"  {left}{content}{RESET}{' ' * pad}{right}")


# ── Clack-style flow primitives ──────────────────────────

def _term_cols() -> int:
    """Width of the terminal; 80 when it is unknown or reported as zero."""
    import shutil
    columns = shutil.get_terminal_size((80, 24)).columns
    return columns if columns > 0 else 80


def bar(text: str = "") -> None:
    """Print a continuation bar line, optionally with text."""
    if text:
        print(f"{_M}{DIM}{_G_BAR}{RESET}{_P_BAR}{text}")
    else:
        print(f"{_M}{DIM}{_G_BAR}{RESET}")


def bar_wrap(text: str, style: str = "") -> None:
    """Print text under the bar, wrapping long lines to fit the terminal.

    Each wrapped line gets its own │ prefix so the left margin is preserved.
    *style* is prepended to each line (e.g. DIM).
    """
    import textwrap
    # "   │  " = 6 visible columns, plus 4 right margin
    usable = max(_term_cols() - 10, 20)
    prefix = f"{_M}{DIM}{_G_BAR}{RESET}{_P_BAR}"
    for logical_line in text.split("\n"):
        wrapped = textwrap.wrap(logical_line, width=usable) if logical_line.strip() else [""]
        for wl in wrapped:
            print(f"{prefix}{style}{wl}{RESET if style else ''}")


def step_start(title: str) -> None:
    print(f"{_M}{BRIGHT_CYAN}{_G_ACTIVE}{RESET}{_P_ACTIVE}{BOLD}{title}{RESET}")


def step_done(title: str) -> None:
    print(f"{_M}{GREEN}{_G_DONE}{RESET}{_P_DONE}{title}")


def step_end(title: str) -> None:
    print(f"{_M}{DIM}{_G_END}{RESET}{_P_END}{title}")


def note(lines: list) -> None:
    for line in lines:
        print(f"{_M}{DIM}{_G_BAR}{RESET}{_P_BAR}{DIM}{line}{RESET}")


def blank() -> None:
    bar()


def success(msg: str) -> str:
    return f"{GREEN}✓{RESET} {msg}"


def error(msg: str) -> str:
    return f"{BRIGHT_RED}✖{RESET} {msg}"


def warn(msg: str) -> str:
    return f"{YELLOW}!{RESET} {msg}"


# ── Raw key reading ─────────────────────────────────────

def clear_lines(n: int) -> None:
    """Move up n lines and clear each one."""
    for _ in range(n):
        sys.stdout.write(f"{MOVE_UP}{CLEAR_LINE}")
    sys.stdout.flush()


def _read_key_timeout(timeout: float) -> str | None:
    """Like _read_key but returns None if no key within *timeout* seconds."""
    import tty
    import termios
    import select as _sel
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd, termios.TCSADRAIN)  # keep typed-ahead keys
        ready, _, _ = _sel.select([fd], [], [], timeout)
        if not ready:
            return None
        return _read_key_raw(fd)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key_raw(fd: int) -> str:
    """Parse one key from fd (already in raw mode).

    Uses os.read to bypass Python's stdin buffer so that select()
    on fd gives accurate results for escape-sequence detection.
    """
    _rd = lambda: os.read(fd, 1).decode("utf-8", errors="replace")
    ch = _rd()
    if ch == "\r" or ch == "\n":
        return "enter"
    if ch == "\x03":
        return "interrupt"
    if ch == "\t":
        return "tab"
    if ch == "\x7f" or ch == "\x08":
        return "backspace"
    if ch == "\x1b":
        import select as _sel
        ready, _, _ = _sel.select([fd], [], [], 0.05)
        if not ready:
            return "escape"
        ch2 = _rd()
        if ch2 == "[":
            ch3 = _rd()
            if ch3 == "A":
                return "up"
            if ch3 == "B":
                return "down"
            if ch3 == "C":
                return "right"
            if ch3 == "D":
                return "left"
            if ch3 == "Z":
                return "shift-tab"
            if ch3 == "5":
                ch4 = _rd()
                if ch4 == "~":
                    return "page-up"
            if ch3 == "6":
                ch4 = _rd()
                if ch4 == "~":
                    return "page-down"
        return "escape"
    return ch


def _read_key() -> str:
    """Block until a key is pressed, then return its name."""
    import tty
    import termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd, termios.TCSADRAIN)  # keep typed-ahead keys
        return _read_key_raw(fd)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── Prompts ──────────────────────────────────────────────

def prompt(label: str, default: str = "") -> str:
    hint = f" {DIM}({default}){RESET}" if default else ""
    try:
        value = input(f"{_M}{BRIGHT_CYAN}{_G_ACTIVE}{RESET}{_P_ACTIVE}{label}{hint} {BRIGHT_CYAN}›{RESET} ")
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return value.strip() or default


def prompt_path(label: str, default: str = "") -> str:
    """Prompt for a filesystem path with Tab completion."""
    if not _IS_TTY:
        return prompt(label, default=default)

    from pathlib import Path as _P
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completion, PathCompleter
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings

    class _SlashPathCompleter(PathCompleter):
        """PathCompleter that appends / to directory completions."""
        def get_completions(self, document, complete_event):
            before = document.text_before_cursor
            for c in super().get_completions(document, complete_event):
                full = before[:len(before) + c.start_position] + c.text
                if _P(full).expanduser().is_dir() and not c.text.endswith("/"):
                    yield Completion(
                        c.text + "/",
                        start_position=c.start_position,
                        display=c.display,
                        display_meta=c.display_meta,
                    )
                else:
                    yield c

    print(f"{_M}{BRIGHT_CYAN}{_G_ACTIVE}{RESET}{_P_ACTIVE}{label}")

    prefix = FormattedText([
        ("", "   "),
        ("fg:gray", "│"),
        ("", "  "),
    ])

    bindings = KeyBindings()

    @bindings.add("enter")
    def _enter(event):
        buf = event.app.current_buffer
        if buf.complete_state and buf.complete_state.current_completion:
            # Dropdown open: select the highlighted completion
            buf.apply_completion(buf.complete_state.current_completion)
            return
        # No dropdown: check if the path is incomplete
        from pathlib import Path as _P
        from prompt_toolkit.completion import CompleteEvent
        text = buf.text.strip()
        if text and _P(text).expanduser().exists():
            # Valid path: submit
            buf.validate_and_handle()
        elif buf.completer:
            # Path doesn't exist yet: try completing
            completions = list(buf.completer.get_completions(
                buf.document, CompleteEvent(text_inserted=False)))
            if len(completions) == 1:
                buf.apply_completion(completions[0])
            elif len(completions) > 1:
                buf.start_completion()
            else:
                buf.validate_and_handle()
        else:
            buf.validate_and_handle()

    session = PromptSession(
        completer=_SlashPathCompleter(expanduser=True),
        key_bindings=bindings,
    )
    try:
        value = session.prompt(prefix, default=default)
    except (EOFError, KeyboardInterrupt):
        return default

    return value.strip() or default


def prompt_tag(label: str) -> str | None:
    """Text input with a Skip option below. Returns string or None.

    First option is a live text field, second is Skip.
    Arrow down/up switches between them. Full inline editing.
    """
    if not _IS_TTY:
        v = prompt(label, default="")
        return v.strip() or None

    buf = ""
    cur = 0  # cursor position in buf
    focus = 0  # 0 = text field, 1 = skip

    def _draw(first: bool = False) -> None:
        lines = 3  # text line + skip line + bar
        if not first:
            clear_lines(lines)

        # Text field line
        if focus == 0:
            before = buf[:cur]
            after = buf[cur:]
            cursor_ch = f"{BRIGHT_CYAN}\u2588{RESET}"
            print(f"{_M}{DIM}{_G_BAR}{RESET}{_P_BAR}{BRIGHT_CYAN}{_G_RADIO_ON}{RESET}{_P_RADIO_ON}{before}{cursor_ch}{after}")
        else:
            print(f"{_M}{DIM}{_G_BAR}{RESET}{_P_BAR}{DIM}{_G_RADIO_OFF}{_P_RADIO_OFF}{buf}{RESET}")

        # Skip line
        if focus == 1:
            print(f"{_M}{DIM}{_G_BAR}{RESET}{_P_BAR}{BRIGHT_CYAN}{_G_RADIO_ON}{RESET}{_P_RADIO_ON}{BOLD}Skip{RESET}")
        else:
            print(f"{_M}{DIM}{_G_BAR}{RESET}{_P_BAR}{DIM}{_G_RADIO_OFF}{_P_RADIO_OFF}Skip{RESET}")

        print(f"{_M}{DIM}{_G_BAR}{RESET}")

    print(f"{_M}{BRIGHT_CYAN}{_G_ACTIVE}{RESET}{_P_ACTIVE}{BOLD}{label}{RESET}")
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.flush()

    try:
        _draw(first=True)

        while True:
            key = _read_key()

            if key == "enter":
                if focus == 1:
                    # Skip
                    clear_lines(3)
                    print(f"{_M}{DIM}{_G_BAR}{RESET}{_P_BAR}{DIM}Skipped{RESET}")
                    return None
                else:
                    # Confirm text
                    clear_lines(3)
                    if buf.strip():
                        print(f"{_M}{DIM}{_G_BAR}{RESET}{_P_BAR}{buf}")
                    else:
                        print(f"{_M}{DIM}{_G_BAR}{RESET}{_P_BAR}{DIM}Skipped{RESET}")
                    return buf.strip() or None
            elif key == "down" and focus == 0:
                focus = 1
                _draw()
            elif key == "up" and focus == 1:
                focus = 0
                _draw()
            elif key == "interrupt":
                clear_lines(3)
                raise KeyboardInterrupt
            elif key == "escape":
                clear_lines(3)
                print(f"{_M}{DIM}{_G_BAR}{RESET}{_P_BAR}{DIM}Skipped{RESET}")
                return None
            elif focus == 0:
                # Text editing
                if key == "left":
                    if cur > 0:
                        cur -= 1
                    _draw()
                elif key == "right":
                    if cur < len(buf):
                        cur += 1
                    _draw()
                elif key == "backspace":
                    if cur > 0:
                        buf = buf[:cur - 1] + buf[cur:]
                        cur -= 1
                    _draw()
                elif len(key) == 1 and key.isprintable():
                    buf = buf[:cur] + key + buf[cur:]
                    cur += 1
                    _draw()
    finally:
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()


def prompt_ghost(label: str, ghost: str) -> str:
    """Prompt with ghost/default text, fully editable with proper cursor.

    Tab fills the ghost into the editable buffer.
    Full cursor, arrows, Home/End, backspace: all work.
    """
    if not _IS_TTY:
        return prompt(label, default=ghost)

    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import ANSI, FormattedText
    from prompt_toolkit.key_binding import KeyBindings

    print(f"{_M}{BRIGHT_CYAN}{_G_ACTIVE}{RESET}{_P_ACTIVE}{label}")

    bindings = KeyBindings()

    @bindings.add("tab")
    def _tab(event):
        buf = event.app.current_buffer
        if not buf.text:
            buf.text = ghost
            buf.cursor_position = len(ghost)

    @bindings.add("enter")
    def _enter(event):
        event.current_buffer.validate_and_handle()

    prefix = FormattedText([
        ("", "   "),
        ("fg:gray", "│"),
        ("", "  "),
    ])
    continuation = FormattedText([
        ("", "   "),
        ("fg:gray", "│"),
        ("", "  "),
    ])

    session = PromptSession(key_bindings=bindings, erase_when_done=True)
    try:
        value = session.prompt(
            prefix,
            placeholder=ANSI(f"{DIM}{ghost}{RESET}"),
            prompt_continuation=continuation,
            multiline=True,
        )
    except (EOFError, KeyboardInterrupt):
        return ghost

    result = value.strip() if value.strip() else ghost

    # erase_when_done=True cleared prompt_toolkit's rendering.
    # Reprint as normal wrapped text.
    bar_wrap(result, style=DIM)

    return result


def select(label: str, options: list, default: int = 0) -> int:
    if not _IS_TTY:
        return _select_fallback(label, options, default)

    selected = default
    n = len(options)

    def _draw(first: bool = False) -> None:
        if not first:
            for _ in range(n + 1):
                sys.stdout.write(f"{MOVE_UP}{CLEAR_LINE}")

        for i, opt in enumerate(options):
            if i == selected:
                print(f"{_M}{DIM}{_G_BAR}{RESET}{_P_BAR}{BRIGHT_CYAN}{_G_RADIO_ON}{RESET}{_P_RADIO_ON}{BOLD}{opt}{RESET}")
            else:
                print(f"{_M}{DIM}{_G_BAR}{RESET}{_P_BAR}{DIM}{_G_RADIO_OFF}{_P_RADIO_OFF}{opt}{RESET}")
        print(f"{_M}{DIM}{_G_BAR}{RESET}")

    print(f"{_M}{BRIGHT_CYAN}{_G_ACTIVE}{RESET}{_P_ACTIVE}{BOLD}{label}{RESET}")
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.flush()

    try:
        _draw(first=True)

        while True:
            key = _read_key()
            if key == "up":
                selected = (selected - 1) % n
                _draw()
            elif key == "down":
                selected = (selected + 1) % n
                _draw()
            elif key == "enter":
                break
            elif key == "interrupt":
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()
        print()
        raise
    finally:
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()

    # Collapse to chosen
    for _ in range(n + 1):
        sys.stdout.write(f"{MOVE_UP}{CLEAR_LINE}")
    print(f"{_M}{DIM}{_G_BAR}{RESET}{_P_BAR}{GREEN}{_G_RADIO_ON}{RESET}{_P_RADIO_ON}{options[selected]}")

    return selected


def _select_fallback(label: str, options: list, default: int) -> int:
    print(f"{_M}{BRIGHT_CYAN}{_G_ACTIVE}{RESET}{_P_ACTIVE}{BOLD}{label}{RESET}")
    for i, opt in enumerate(options):
        if i == default:
            print(f"{_M}{DIM}{_G_BAR}{RESET}{_P_BAR}{BRIGHT_CYAN}{_G_RADIO_ON}{RESET}{_P_RADIO_ON}{i + 1}. {opt}")
        else:
            print(f"{_M}{DIM}{_G_BAR}{RESET}{_P_BAR}{DIM}{_G_RADIO_OFF}{_P_RADIO_OFF}{i + 1}. {opt}{RESET}")
    try:
        value = input(f"{_M}{DIM}{_G_BAR}{RESET}{_P_BAR}Choice [{default + 1}]: ")
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    value = value.strip()
    if not value:
        return default
    try:
        idx = int(value) - 1
        if 0 <= idx < len(options):
            return idx
    except ValueError:
        pass
    return default


# ── Table helpers ────────────────────────────────────────

def table_header(*cols: tuple) -> None:
    parts = []
    for label, width in cols:
        parts.append(f"{BOLD}{BRIGHT_WHITE}{label:<{width}}{RESET}")
    print(f"  {''.join(parts)}")
    total = sum(w for _, w in cols)
    print(f"  {DIM}{'─' * total}{RESET}")


def table_row(values: list, widths: list) -> None:
    parts = []
    for val, w in zip(values, widths):
        vlen = _visible_len(val)
        pad = max(w - vlen, 0)
        parts.append(f"{val}{' ' * pad}")
    print(f"  {''.join(parts)}")
