#!/usr/bin/env python3
"""Turn an agent's JSON event stream into a readable transcript.

Reads JSON lines from stdin and prints one readable block per event. Handles
the event shapes of Claude Code (``--output-format stream-json``) and Codex
(``codex exec --json``), dispatched on ``type``. Standard library only.

Optional trajectory capture (silent side-effect: does not change stdout):
  ``--agent {claude-code,codex} --traj-dir PATH --phase {main,test}``
  Each invocation writes its raw events to ``<traj-dir>/turn_NNNN_raw.jsonl``
  and merges per-turn stats into ``<traj-dir>/stats.json`` with a unified
  schema (input_tokens, output_tokens, num_turns, duration_s, tool_uses,
  cost_usd) so Claude and Codex share the same shape.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


# ── ANSI colors ────────────────────────────────────────────────

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[95m"
RED = "\033[31m"
GRAY = "\033[90m"


def _truncate(s: str, n: int = 200) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _multiline(s: str, max_lines: int = 40, max_chars: int = 4000) -> str:
    """Preserve line breaks but cap overall size so the log doesn't explode."""
    if len(s) > max_chars:
        s = s[:max_chars] + "\n…[truncated]"
    lines = s.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"…[+{len(lines) - max_lines} more lines]"]
    return "\n".join(lines)


def _format_tool_input(name: str, inp: dict) -> str:
    """Short summary of what a tool call is doing."""
    if name == "Read":
        return inp.get("file_path", "?")
    if name in ("Write", "Edit"):
        return inp.get("file_path", "?")
    if name == "Bash":
        cmd = inp.get("command", "?")
        return _truncate(cmd, 120)
    if name == "Glob":
        return inp.get("pattern", "?")
    if name == "Grep":
        return f"{inp.get('pattern', '?')}  in {inp.get('path', '.')}"
    if name == "TodoWrite":
        todos = inp.get("todos", [])
        return f"{len(todos)} todo(s)"
    return _truncate(json.dumps(inp, ensure_ascii=False), 120)


def handle(event: dict) -> None:
    t = event.get("type")

    if t == "system" and event.get("subtype") == "init":
        model = event.get("model", "?")
        cwd = event.get("cwd", "?")
        print(f"{BOLD}{CYAN}● session started{RESET}  "
              f"{DIM}model={model}  cwd={cwd}{RESET}", flush=True)
        return

    if t == "assistant":
        msg = event.get("message", {})
        for block in msg.get("content", []):
            btype = block.get("type")
            if btype == "text":
                text = block.get("text", "").strip()
                if text:
                    # Full assistant text, preserved with indentation
                    print(f"{MAGENTA}● assistant{RESET}", flush=True)
                    for line in text.splitlines():
                        print(f"  {line}", flush=True)
            elif btype == "tool_use":
                name = block.get("name", "?")
                summary = _format_tool_input(name, block.get("input", {}))
                print(f"{YELLOW}● {name:<10}{RESET} {DIM}{summary}{RESET}", flush=True)
            elif btype == "thinking":
                text = block.get("thinking", "").strip()
                if text:
                    print(f"{GRAY}● thinking{RESET}", flush=True)
                    for line in _multiline(text, max_lines=20, max_chars=1500).splitlines():
                        print(f"  {GRAY}{line}{RESET}", flush=True)
        return

    if t == "user":
        msg = event.get("message", {})
        for block in msg.get("content", []):
            if block.get("type") == "tool_result":
                content = block.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        c.get("text", "") for c in content if isinstance(c, dict)
                    )
                is_err = block.get("is_error", False)
                body = _multiline(str(content).rstrip(),
                                  max_lines=40, max_chars=4000)
                if is_err:
                    print(f"{RED}● tool err{RESET}", flush=True)
                    for line in body.splitlines():
                        print(f"  {RED}{line}{RESET}", flush=True)
                else:
                    # Empty results are common (silent success): say so
                    if not body.strip():
                        print(f"{GREEN}● result{RESET}  {DIM}(ok, no output){RESET}",
                              flush=True)
                    else:
                        print(f"{GREEN}● result{RESET}", flush=True)
                        for line in body.splitlines():
                            print(f"  {DIM}{line}{RESET}", flush=True)
        return

    if t == "result":
        status = event.get("subtype", "success")
        cost = event.get("total_cost_usd")
        turns = event.get("num_turns")
        pieces = [f"turns={turns}" if turns is not None else ""]
        if cost is not None:
            pieces.append(f"cost=${cost:.4f}")
        suffix = "  " + "  ".join(p for p in pieces if p) if any(pieces) else ""
        color = GREEN if status == "success" else RED
        print(f"{color}{BOLD}● done{RESET}  {DIM}{status}{suffix}{RESET}",
              flush=True)
        return

    # ── Codex (codex exec --json) event shapes ───────────────────
    handle_codex(event, t)


def _fmt_command(cmd) -> str:
    """One-line form of a Codex command, without its ``bash -lc`` wrapper."""
    if isinstance(cmd, list):
        if len(cmd) >= 3 and cmd[0] in ("bash", "sh", "/bin/bash", "/bin/sh") and cmd[1] in ("-lc", "-c"):
            return _truncate(str(cmd[2]), 160)
        return _truncate(" ".join(str(c) for c in cmd), 160)
    text = str(cmd)
    for prefix in ("/bin/bash -lc ", "bash -lc ", "/bin/sh -c ", "sh -c "):
        if text.startswith(prefix):
            text = text[len(prefix):]
            if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
                text = text[1:-1]
            break
    return _truncate(text, 160)


def handle_codex(event: dict, t: str | None) -> None:
    if t in ("session.created", "thread.started", "session_meta"):
        sid = (
            event.get("session_id")
            or event.get("thread_id")
            or event.get("payload", {}).get("id")
            or "?"
        )
        print(f"{BOLD}{CYAN}● session started{RESET}  {DIM}id={sid}{RESET}",
              flush=True)
        return

    if t == "error":
        msg = event.get("message", str(event))
        print(f"{RED}● error{RESET}  {RED}{_truncate(msg, 300)}{RESET}",
              flush=True)
        return

    if t == "event_msg":
        payload = event.get("payload", {}) or {}
        e = payload.get("type", "")

        if e == "agent_message":
            text = (payload.get("message") or "").strip()
            if text:
                print(f"{MAGENTA}● assistant{RESET}", flush=True)
                for line in text.splitlines():
                    print(f"  {line}", flush=True)
            return

        if e == "agent_reasoning":
            text = (payload.get("text") or payload.get("message") or "").strip()
            if text:
                print(f"{GRAY}● thinking{RESET}", flush=True)
                for line in _multiline(text, max_lines=20, max_chars=1500).splitlines():
                    print(f"  {GRAY}{line}{RESET}", flush=True)
            return

        if e == "exec_command":
            cmd = _fmt_command(payload.get("command", "?"))
            print(f"{YELLOW}● Bash      {RESET} {DIM}{cmd}{RESET}", flush=True)
            return

        if e in ("error", "stream_error"):
            msg = payload.get("message", str(payload))
            print(f"{RED}● error{RESET}  {RED}{_truncate(msg, 300)}{RESET}",
                  flush=True)
            return

        # token_count and other meta events: stay quiet
        return

    if t == "item.started":
        item = event.get("item", {}) or {}
        if (item.get("type") or item.get("item_type", "")) == "command_execution":
            cmd = _fmt_command(item.get("command", "?"))
            print(f"{YELLOW}● Bash      {RESET} {DIM}{cmd}{RESET}", flush=True)
        return

    if t == "turn.failed":
        error = event.get("error") or {}
        msg = error.get("message") if isinstance(error, dict) else str(error)
        print(f"{RED}● error{RESET}  {RED}{_truncate(msg or str(event), 300)}{RESET}", flush=True)
        return

    if t == "item.completed":
        item = event.get("item", {}) or {}
        item_type = item.get("type") or item.get("item_type", "")

        if item_type in ("assistant_message", "agent_message"):
            text = (item.get("text") or "").strip()
            if text:
                print(f"{MAGENTA}● assistant{RESET}", flush=True)
                for line in text.splitlines():
                    print(f"  {line}", flush=True)
            return

        if item_type == "reasoning":
            text = (item.get("text") or "").strip()
            if text:
                print(f"{GRAY}● thinking{RESET}", flush=True)
                for line in _multiline(text, max_lines=20, max_chars=1500).splitlines():
                    print(f"  {GRAY}{line}{RESET}", flush=True)
            return

        if item_type == "error":
            msg = item.get("message") or str(item)
            print(f"{RED}● error{RESET}  {RED}{_truncate(msg, 300)}{RESET}", flush=True)
            return

        if item_type == "command_execution":
            exit_code = item.get("exit_code")
            stdout = item.get("aggregated_output") or item.get("stdout") or ""
            stderr = item.get("stderr") or ""
            body = stdout if stdout else stderr
            body = _multiline(str(body).rstrip(), max_lines=40, max_chars=4000)
            if exit_code not in (None, 0):
                print(f"{RED}● tool err{RESET}  {DIM}exit={exit_code}{RESET}",
                      flush=True)
                for line in body.splitlines():
                    print(f"  {RED}{line}{RESET}", flush=True)
            else:
                if not body.strip():
                    print(f"{GREEN}● result{RESET}  {DIM}(ok, no output){RESET}",
                          flush=True)
                else:
                    print(f"{GREEN}● result{RESET}", flush=True)
                    for line in body.splitlines():
                        print(f"  {DIM}{line}{RESET}", flush=True)
            return

        if item_type == "function_call":
            name = item.get("name", "?")
            args = item.get("arguments") or item.get("args") or {}
            if isinstance(args, str):
                summary = _truncate(args, 120)
            else:
                summary = _truncate(json.dumps(args, ensure_ascii=False), 120)
            print(f"{YELLOW}● {name:<10}{RESET} {DIM}{summary}{RESET}", flush=True)
            return

        if item_type == "file_change":
            changes = item.get("changes")
            if not isinstance(changes, list):
                changes = [item]
            for change in changes:
                if not isinstance(change, dict):
                    continue
                path = change.get("path") or change.get("file_path") or "?"
                kind = change.get("kind") or change.get("change_kind") or "edit"
                print(f"{YELLOW}● {kind:<10}{RESET} {DIM}{path}{RESET}", flush=True)
            return

        if item_type == "mcp_tool_call":
            name = f"{item.get('server', '?')}.{item.get('tool', '?')}"
            print(f"{YELLOW}● {name:<10}{RESET}", flush=True)
            return

        if item_type == "web_search":
            print(f"{YELLOW}● search    {RESET} {DIM}{_truncate(str(item.get('query', '')), 120)}{RESET}",
                  flush=True)
            return

        return

    if t == "turn.completed":
        usage = event.get("usage", {}) or {}
        pieces = []
        if usage.get("input_tokens") is not None:
            pieces.append(f"in={usage['input_tokens']}")
        if usage.get("output_tokens") is not None:
            pieces.append(f"out={usage['output_tokens']}")
        suffix = "  " + "  ".join(pieces) if pieces else ""
        print(f"{GREEN}● turn done{RESET}  {DIM}{suffix.strip()}{RESET}",
              flush=True)
        return

    if t == "task.completed" or t == "task_complete":
        print(f"{GREEN}{BOLD}● done{RESET}", flush=True)
        return


# ── Stats accumulation (unified schema across Claude Code / Codex) ──
#
# One schema regardless of agent:
#   input_tokens  : total tokens the model saw (Claude: base+cache_read+cache_create;
#                    Codex: input_tokens as reported).
#   output_tokens : total output produced (includes any reasoning tokens).
#   num_turns     : agent's self-reported inner turn count (may be 0 if absent).
#   tool_uses     : {tool_name: count}.
#   cost_usd      : Claude only; 0 for Codex.
# Plus per-turn metadata: phase, agent, session_id, errors, duration_s.


def _new_turn_stats(phase: str, agent: str) -> dict:
    return {
        "phase": phase,
        "agent": agent,
        "session_id": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "num_turns": 0,
        "duration_s": 0.0,
        "cost_usd": 0.0,
        "tool_uses": {},
        "errors": [],
    }


def _tool_bump(stats: dict, name: str) -> None:
    stats["tool_uses"][name] = stats["tool_uses"].get(name, 0) + 1


def _update_claude(event: dict, stats: dict) -> None:
    t = event.get("type")

    if t == "system" and event.get("subtype") == "init":
        stats["session_id"] = event.get("session_id") or stats["session_id"]
        return

    if t == "assistant":
        msg = event.get("message", {}) or {}
        for block in msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                _tool_bump(stats, block.get("name", "unknown"))
        return

    if t == "result":
        # Authoritative final counts for this CLI invocation.
        stats["session_id"] = event.get("session_id") or stats["session_id"]
        usage = event.get("usage", {}) or {}
        base = usage.get("input_tokens", 0) or 0
        cache_read = usage.get("cache_read_input_tokens", 0) or 0
        cache_create = usage.get("cache_creation_input_tokens", 0) or 0
        # Unified: total input the model saw.
        stats["input_tokens"] = base + cache_read + cache_create
        stats["output_tokens"] = usage.get("output_tokens", 0) or 0
        if event.get("total_cost_usd") is not None:
            stats["cost_usd"] = float(event["total_cost_usd"])
        elif event.get("cost_usd") is not None:
            stats["cost_usd"] = float(event["cost_usd"])
        stats["num_turns"] = event.get("num_turns", 0) or 0
        if event.get("is_error"):
            msg = event.get("result") or ""
            if msg:
                stats["errors"].append(str(msg))
        return

    if t == "error":
        err = event.get("error") or {}
        msg = err.get("message") if isinstance(err, dict) else None
        stats["errors"].append(msg or str(err or event))


def _update_codex(event: dict, stats: dict) -> None:
    t = event.get("type")

    if t in ("session.created", "thread.started", "session_meta"):
        stats["session_id"] = (
            event.get("session_id")
            or event.get("thread_id")
            or (event.get("payload", {}) or {}).get("id")
            or stats["session_id"]
        )
        return

    if t == "event_msg":
        payload = event.get("payload", {}) or {}
        ptype = payload.get("type", "")
        if ptype == "exec_command":
            _tool_bump(stats, "shell")
        elif ptype == "token_count":
            info = payload.get("info") or {}
            total = info.get("total_token_usage") or {}
            if total:
                stats["input_tokens"] = total.get("input_tokens", stats["input_tokens"])
                stats["output_tokens"] = total.get("output_tokens", stats["output_tokens"])
        elif ptype in ("error", "stream_error"):
            stats["errors"].append(payload.get("message", str(payload)))
        return

    if t == "item.completed":
        item = event.get("item", {}) or {}
        item_type = item.get("type") or item.get("item_type", "")
        if item_type == "command_execution":
            _tool_bump(stats, "shell")
        elif item_type == "function_call":
            _tool_bump(stats, item.get("name", "unknown"))
        elif item_type == "file_change":
            _tool_bump(stats, "file_edit")
        elif item_type == "mcp_tool_call":
            _tool_bump(stats, f"{item.get('server', 'mcp')}.{item.get('tool', 'unknown')}")
        elif item_type == "web_search":
            _tool_bump(stats, "web_search")
        elif item_type == "error":
            stats["errors"].append(str(item.get("message") or item))
        return

    if t == "turn.failed":
        error = event.get("error") or {}
        stats["errors"].append(str(error.get("message") if isinstance(error, dict) else error))
        return

    if t == "turn.completed":
        usage = event.get("usage", {}) or {}
        stats["input_tokens"] = usage.get("input_tokens", stats["input_tokens"])
        stats["output_tokens"] = usage.get("output_tokens", stats["output_tokens"])
        stats["num_turns"] = (stats.get("num_turns") or 0) + 1
        return

    if t == "error":
        stats["errors"].append(event.get("message", str(event)))


def _finalize_trajectory(traj_dir: str, turn_stats: dict) -> None:
    """Persist this invocation's raw log pointer + merge into stats.json.

    raw_path is already populated turn by turn; here we only touch stats.json.
    Schema:
      {
        "scaffold": "claude-code" | "codex",
        "turns": [ <turn_stats>, ... ],
        "cumulative_stats": { merged totals }
      }
    """
    os.makedirs(traj_dir, exist_ok=True)
    stats_path = os.path.join(traj_dir, "stats.json")

    try:
        with open(stats_path, "r") as f:
            doc = json.load(f)
        if not isinstance(doc, dict):
            doc = {}
    except (FileNotFoundError, json.JSONDecodeError):
        doc = {}

    turns = doc.get("turns", [])
    if not isinstance(turns, list):
        turns = []
    turn_stats["turn"] = len(turns) + 1
    turns.append(turn_stats)

    # Merge cumulative totals.
    cum = {
        "input_tokens": 0,
        "output_tokens": 0,
        "num_turns": 0,
        "duration_s": 0.0,
        "cost_usd": 0.0,
        "tool_uses": {},
    }
    for rec in turns:
        cum["input_tokens"] += int(rec.get("input_tokens", 0) or 0)
        cum["output_tokens"] += int(rec.get("output_tokens", 0) or 0)
        cum["num_turns"] += int(rec.get("num_turns", 0) or 0)
        cum["duration_s"] += float(rec.get("duration_s", 0.0) or 0.0)
        cum["cost_usd"] += float(rec.get("cost_usd", 0.0) or 0.0)
        for name, count in (rec.get("tool_uses") or {}).items():
            cum["tool_uses"][name] = cum["tool_uses"].get(name, 0) + int(count)

    doc["scaffold"] = turn_stats["agent"]
    doc["turns"] = turns
    doc["cumulative_stats"] = cum

    # Atomic write via temp + rename.
    tmp_path = stats_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp_path, stats_path)


def _next_turn_path(traj_dir: str) -> str:
    """Pick <traj-dir>/turn_NNNN_raw.jsonl with the next available number."""
    os.makedirs(traj_dir, exist_ok=True)
    existing = [n for n in os.listdir(traj_dir)
                if n.startswith("turn_") and n.endswith("_raw.jsonl")]
    return os.path.join(traj_dir, f"turn_{len(existing) + 1:04d}_raw.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--agent", choices=("claude-code", "codex"), default=None)
    ap.add_argument("--traj-dir", default=None)
    ap.add_argument("--phase", default="main")
    args, _unknown = ap.parse_known_args()

    traj_enabled = bool(args.traj_dir and args.agent)
    stats = _new_turn_stats(args.phase, args.agent) if traj_enabled else None
    raw_fh = None
    start = time.monotonic()
    if traj_enabled:
        raw_fh = open(_next_turn_path(args.traj_dir), "w")

    try:
        for line in sys.stdin:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                # Not JSON: print as-is (could be a plain error from the CLI)
                print(stripped, flush=True)
                continue

            # Pretty-print.
            try:
                handle(event)
            except Exception as e:
                print(f"{RED}[format_stream error: {e}]{RESET}  {stripped}",
                      flush=True)

            # Trajectory side-effects.
            if traj_enabled:
                try:
                    raw_fh.write(stripped + "\n")
                    raw_fh.flush()
                    if args.agent == "claude-code":
                        _update_claude(event, stats)
                    else:
                        _update_codex(event, stats)
                except Exception:
                    # Never let trajectory capture crash the pretty-printer.
                    pass
    finally:
        if raw_fh is not None:
            raw_fh.close()
        if traj_enabled:
            stats["duration_s"] = round(time.monotonic() - start, 3)
            try:
                _finalize_trajectory(args.traj_dir, stats)
            except Exception as e:
                # Keep stdout clean; emit a small stderr note so the user
                # knows trajectory wasn't captured if something went wrong.
                sys.stderr.write(f"[format_stream] trajectory write failed: {e}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
