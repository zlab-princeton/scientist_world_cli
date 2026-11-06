"""The transcript printer and trajectory recorder that sits behind the agent."""

import io
import json

from sciworld import format_stream


def feed(monkeypatch, capsys, lines, *argv):
    monkeypatch.setattr("sys.stdin", io.StringIO("\n".join(lines) + "\n"))
    monkeypatch.setattr("sys.argv", ["format_stream", *argv])
    assert format_stream.main() == 0
    return capsys.readouterr().out


def test_claude_events(monkeypatch, capsys, tmp_path):
    events = [
        {"type": "system", "subtype": "init", "model": "m", "cwd": "/w", "session_id": "s"},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Reading the problem."},
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls -la"}},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "/w/a.py"}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": "a.py"},
                                                 {"type": "tool_result", "content": "boom", "is_error": True}]}},
        {"type": "result", "subtype": "success", "session_id": "s", "num_turns": 3, "total_cost_usd": 0.5,
         "usage": {"input_tokens": 10, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 5,
                   "output_tokens": 7}},
    ]
    traj = tmp_path / "trajectory"
    out = feed(monkeypatch, capsys, [json.dumps(e) for e in events] + ["plain text line"],
               "--agent", "claude-code", "--traj-dir", str(traj), "--phase", "main")
    for expected in ("session started", "Reading the problem.", "ls -la", "/w/a.py", "boom",
                     "turns=3", "cost=$0.5000", "plain text line"):
        assert expected in out
    assert len((traj / "turn_0001_raw.jsonl").read_text().splitlines()) == 4
    stats = json.loads((traj / "stats.json").read_text())
    turn = stats["turns"][0]
    assert (turn["input_tokens"], turn["output_tokens"], turn["num_turns"]) == (115, 7, 3)
    assert turn["tool_uses"] == {"Bash": 1, "Read": 1} and turn["phase"] == "main" and turn["session_id"] == "s"
    assert stats["scaffold"] == "claude-code" and stats["cumulative_stats"]["cost_usd"] == 0.5


def test_codex_events_accumulate_over_turns(monkeypatch, capsys, tmp_path):
    events = [
        {"type": "thread.started", "thread_id": "t"},
        {"type": "item.completed", "item": {"type": "reasoning", "text": "Thinking about parity."}},
        {"type": "item.completed", "item": {"type": "command_execution", "command": "ls", "exit_code": 0,
                                            "aggregated_output": "file.txt"}},
        {"type": "item.completed", "item": {"type": "command_execution", "exit_code": 2, "aggregated_output": "bad"}},
        {"type": "item.completed", "item": {"type": "file_change", "changes": [{"path": "m.py", "kind": "add"}]}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "All done."}},
        {"type": "turn.completed", "usage": {"input_tokens": 50, "output_tokens": 9}},
    ]
    traj = tmp_path / "trajectory"
    for phase in ("main", "test"):
        out = feed(monkeypatch, capsys, [json.dumps(e) for e in events],
                   "--agent", "codex", "--traj-dir", str(traj), "--phase", phase)
    for expected in ("session started", "Thinking about parity.", "file.txt", "exit=2", "m.py", "All done.", "in=50"):
        assert expected in out
    stats = json.loads((traj / "stats.json").read_text())
    assert [t["phase"] for t in stats["turns"]] == ["main", "test"]
    assert stats["cumulative_stats"]["input_tokens"] == 100
    assert stats["cumulative_stats"]["tool_uses"] == {"shell": 4, "file_edit": 2}
    assert sorted(p.name for p in traj.glob("turn_*")) == ["turn_0001_raw.jsonl", "turn_0002_raw.jsonl"]


def test_without_a_trajectory_directory_nothing_is_written(monkeypatch, capsys, tmp_path):
    out = feed(monkeypatch, capsys, ['{"type": "error", "message": "oops"}', "", "{broken json"])
    assert "oops" in out and "{broken json" in out
    assert list(tmp_path.iterdir()) == [tmp_path / "home"]
