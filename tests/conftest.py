"""Shared fixtures. Every test runs with a throwaway HOME and SCIWORLD_HOME,
no GitHub token, and no network unless it is marked ``network``."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

ARXIV_ID = "2401.00001"

# Tests replace subprocess.run; the fixture teardown below needs the real one.
_run = subprocess.run


def pytest_collection_modifyitems(config, items):
    if os.environ.get("SCIWORLD_TEST_NETWORK") == "1":
        return
    skip = pytest.mark.skip(reason="set SCIWORLD_TEST_NETWORK=1 to run tests that reach github.com")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Point every piece of user state at the test's temporary directory.

    tmux and screen get private servers too, so no test can see, create or
    end a session of the person running the tests.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SCIWORLD_HOME", str(tmp_path / "sciworld_home"))
    for var in ("SCIWORLD_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN",
                "SCIWORLD_DATA_DIR", "SCIWORLD_ORG",
                "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
                "CLAUDE_CODE_OAUTH_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL",
                "TMUX", "STY"):
        monkeypatch.delenv(var, raising=False)
    import sciworld.github as github
    import sciworld.launcher as launcher
    monkeypatch.setattr(github, "_token_from_gh", lambda: None)
    monkeypatch.setattr(github, "_REJECTED_TOKENS", set())
    # A healthy launch is watched for this long before `sw research` returns.
    monkeypatch.setattr(launcher, "LAUNCH_SETTLE_SECONDS", 0.3)
    # Generated container commands must not depend on the machine running
    # the tests. The tests of real runtimes look at the machine again.
    monkeypatch.setattr(launcher, "selinux_enforcing", lambda: False)

    # Socket paths have a short length limit, so this directory is not under
    # tmp_path.
    base = "/tmp" if os.access("/tmp", os.W_OK) else None
    sockets = Path(tempfile.mkdtemp(prefix="swt-", dir=base))
    (sockets / "screen").mkdir(mode=0o700)
    monkeypatch.setenv("TMUX_TMPDIR", str(sockets))
    monkeypatch.setenv("SCREENDIR", str(sockets / "screen"))
    yield tmp_path
    private = dict(os.environ, TMUX_TMPDIR=str(sockets), SCREENDIR=str(sockets / "screen"))
    if any((sockets).glob("tmux-*")):
        _run(["tmux", "kill-server"], capture_output=True, env=private)
    if any((sockets / "screen").iterdir()):
        listing = _run(["screen", "-ls"], capture_output=True, text=True, env=private).stdout
        for line in listing.splitlines():
            fields = line.split()
            if fields and fields[0].partition(".")[0].isdigit():
                _run(["screen", "-S", fields[0], "-X", "quit"], capture_output=True, env=private)
    shutil.rmtree(sockets, ignore_errors=True)


def _write(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip("\n"))
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


EVALUATOR = '''
    """Scores a prediction file. Shared by the train and the test slice."""
    import json
    import sys

    from eval.metrics import accuracy


    def main(split, predictions, labels, out):
        preds = json.load(open(predictions))
        gold = json.load(open(labels))
        scores = {"experiments": {"main": {"results": {"ours": {
            "accuracy": accuracy(preds, gold), "n": len(gold)}}}}}
        json.dump(scores, open(out, "w"), indent=2)
        print(f"{split}: wrote {out}")
'''


def build_environment(root: Path) -> Path:
    """Create a small but complete environment repository at *root*.

    It mirrors the layout of a real one: a briefing, a scoring contract, a
    visible and a held-out evaluator that share a module, scripts written for
    the container path /home/user, the paper's method, and builder leftovers
    that must never reach a workspace.
    """
    _write(root / "briefing/problem.md", "# Problem\n\nPredict the parity of an integer.\n")
    _write(root / "briefing/evaluation.md", "# Evaluation\n\n## Metrics\n\nAccuracy.\n")
    _write(root / "briefing/method.md", "# Method\n\n## Approach\n\nThe paper uses a lookup table.\n")
    _write(root / "briefing/overview.md",
           "- **Title:** Parity Is All You Need\n- **Domain:** Toy\n- **TL;DR:** A toy task.\n\n"
           "## Short Summary\n\nA toy environment used by the tests.\n")
    for name in ("TARGETS", "CONSTRAINTS", "DIRECTION"):
        _write(root / f"scoring/{name}.md", f"# {name.title()}\n\nText of {name}.\n")
    _write(root / "scoring/EXPERIMENTS.md", "## `main`\n\nParity accuracy on held-in numbers.\n")
    reference = {"experiments": {"main": {
        "weight": 1.0, "primary_metric": "accuracy",
        "metrics": {"accuracy": {"higher_is_better": True, "coefficient": 1.0}},
        "results": {"paper_method": {"type": "proposed"}, "majority": {"type": "baseline"}},
    }}}
    _write(root / "scoring/reference.json", json.dumps(reference, indent=2))
    for split, value in (("train", 0.5), ("test", 0.6)):
        scores = {"experiments": {"main": {"results": {
            "paper_method": {"accuracy": value}, "majority": {"accuracy": 0.25}}}}}
        _write(root / f"scoring/scores_{split}.json", json.dumps(scores, indent=2))
    _write(root / "scoring/scores.json", "{}")

    _write(root / "eval/__init__.py", "")
    _write(root / "eval/metrics.py", '''
        def accuracy(preds, gold):
            hits = sum(1 for key, label in gold.items() if preds.get(key) == label)
            return hits / max(len(gold), 1)
    ''')
    _write(root / "eval/common.py", EVALUATOR)
    _write(root / "eval/unused.py", "RAISES = 'this module is imported by nobody'\n")
    for split in ("train", "test"):
        _write(root / f"eval/{split}/__init__.py", "")
        _write(root / f"eval/{split}/evaluate.py", f'''
            import sys

            from eval.common import main

            if __name__ == "__main__":
                main("{split}", sys.argv[1], "/home/user/eval/{split}/labels.json", sys.argv[2])
        ''')
        labels = {str(i): i % 2 for i in (range(0, 8) if split == "train" else range(100, 110))}
        _write(root / f"eval/{split}/labels.json", json.dumps(labels))

    for split in ("train", "test"):
        _write(root / f"scripts/evaluate_{split}.sh", f'''
            #!/bin/bash
            set -e
            cd /home/user
            mkdir -p scoring
            python3 -m eval.{split}.evaluate /home/user/method/predictions_{split}.json \\
                /home/user/scoring/scores_{split}.json
        ''', executable=True)
    _write(root / "scripts/baseline.sh", "#!/bin/bash\ncd /home/user\necho baseline\n", executable=True)
    _write(root / "scripts/download.sh", "#!/bin/bash\ncd /home/user\necho nothing to download\n", executable=True)
    for name in ("run.sh", "method.sh", "reproduce.sh", "evaluate.sh"):
        _write(root / f"scripts/{name}", "#!/bin/bash\nbash scripts/evaluate_test.sh\n", executable=True)

    _write(root / "method/predict.py", "def predict(n):\n    return n % 2\n")
    _write(root / "baseline/majority.py", "def predict(n):\n    return 0\n")
    _write(root / "data/numbers.json", json.dumps(list(range(8))))
    _write(root / "data/loader.py", "PATH = '/home/user/data/numbers.json'\n")
    _write(root / "environment/container.def", "Bootstrap: docker\nFrom: python:3.11-slim\n")
    _write(root / "environment/setup.sh", "pip install nothing\n")
    _write(root / "environment/container.sif", "binary\n")

    # Builder leftovers: none of this belongs in a workspace.
    _write(root / "paper/paper.md", "The paper.\n")
    _write(root / "checkpoints/model.bin", "weights\n")
    _write(root / "jobs/1/stdout.log", "log\n")
    _write(root / "PROGRESS.md", "notes\n")
    _write(root / "README.md", "- **TL;DR:** A toy task.\n")
    return root


@pytest.fixture
def env_source(tmp_path) -> Path:
    """An environment repository on disk (not a git repository)."""
    return build_environment(tmp_path / "repo")


@pytest.fixture
def data_dir(tmp_path) -> Path:
    """A data directory in which the environment is already downloaded."""
    root = tmp_path / "data"
    build_environment(root / "envs" / ARXIV_ID / "source")
    (root / "containers").mkdir(parents=True)
    return root


@pytest.fixture
def configured(data_dir):
    """sciworld set up for venv mode with the nohup session manager."""
    from sciworld import settings as config
    cfg = dict(config.DEFAULT_CONFIG)
    cfg.update(data_dir=str(data_dir), container="venv", session="nohup")
    config.save_config(cfg)
    return cfg


FAKE_AGENT = r'''
#!{python}
"""Stand-in for an agent CLI. Reads the prompt from stdin, acts on the
workspace according to which prompt it got, and prints the event stream the
real CLI would print."""
import fcntl, json, os, subprocess, sys
from pathlib import Path

name = Path(sys.argv[0]).name
argv = sys.argv[1:]
prompt = sys.stdin.read()
cwd = Path.cwd()
record = Path(os.environ["FAKE_AGENT_RECORD"])
# Several runs may record at the same moment: one writer at a time, and the
# file is replaced in one step so a reader never sees it half written.
with open(str(record) + ".lock", "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    calls = json.loads(record.read_text()) if record.exists() else []
    calls.append({{"name": name, "argv": argv, "prompt": prompt, "cwd": str(cwd),
                  "env": {{k: os.environ.get(k) for k in ("VIRTUAL_ENV", "ANTHROPIC_API_KEY", "MY_FORWARDED")}}}})
    scratch = record.with_name(f"{{record.name}}.{{os.getpid()}}.tmp")
    scratch.write_text(json.dumps(calls))
    os.replace(scratch, record)

fail_first = int(os.environ.get("FAKE_AGENT_FAIL_FIRST", "0"))
if len(calls) <= fail_first:
    print(json.dumps({{"type": "error", "message": "simulated crash"}}))
    sys.exit(3)

# FAKE_AGENT_WAIT_FOR: keep "working" until that file appears.
gate = os.environ.get("FAKE_AGENT_WAIT_FOR")
if gate:
    import time
    deadline = time.time() + 300
    while not os.path.exists(gate) and time.time() < deadline:
        time.sleep(0.1)

def emit(event):
    print(json.dumps(event), flush=True)

if name == "claude":
    emit({{"type": "system", "subtype": "init", "model": "fake-model", "cwd": str(cwd), "session_id": "s-1"}})
else:
    emit({{"type": "thread.started", "thread_id": "t-1"}})

def say(text):
    if name == "claude":
        emit({{"type": "assistant", "message": {{"content": [{{"type": "text", "text": text}}]}}}})
    else:
        emit({{"type": "item.completed", "item": {{"type": "agent_message", "text": text}}}})

def set_done():
    (cwd / "status.json").write_text(json.dumps({{"done": True}}))

if "test slice has been added" in prompt:
    preds = {{str(i): i % 2 for i in range(100, 110)}}
    (cwd / "method" / "predictions_test.json").write_text(json.dumps(preds)) if os.access(cwd / "method", os.W_OK) else None
    result = subprocess.run(["bash", "scripts/evaluate_test.sh"], capture_output=True, text=True)
    say("test evaluation: " + (result.stdout + result.stderr).strip())
    (cwd / "reflection.md").write_text("It works because parity is n mod 2.\n")
    if os.environ.get("FAKE_AGENT_TEST_SETS_DONE", "1") == "1":
        set_done()
elif "Continue your work" in prompt:
    say("continuing")
    set_done()
else:
    (cwd / "method").mkdir(exist_ok=True)
    (cwd / "method" / "predict.py").write_text("def predict(n):\n    return n % 2\n")
    (cwd / "method" / "predictions_train.json").write_text(json.dumps({{str(i): i % 2 for i in range(8)}}))
    (cwd / "method" / "predictions_test.json").write_text(json.dumps({{str(i): i % 2 for i in range(100, 110)}}))
    result = subprocess.run(["bash", "scripts/evaluate_train.sh"], capture_output=True, text=True)
    say("train evaluation: " + (result.stdout + result.stderr).strip())
    (cwd / "idea.md").write_text("Use n mod 2.\n")
    if os.environ.get("FAKE_AGENT_DONE_FIRST", "1") == "1":
        set_done()

if name == "claude":
    emit({{"type": "result", "subtype": "success", "session_id": "s-1", "num_turns": 2,
          "total_cost_usd": 0.01, "usage": {{"input_tokens": 10, "output_tokens": 5}}}})
else:
    emit({{"type": "turn.completed", "usage": {{"input_tokens": 10, "output_tokens": 5}}}})
'''


@pytest.fixture
def fake_agent(tmp_path, monkeypatch):
    """Put fake ``claude`` and ``codex`` executables first on PATH.

    Returns the path of the JSON file in which they record their calls.
    """
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    for name in ("claude", "codex"):
        path = bindir / name
        path.write_text(FAKE_AGENT.lstrip("\n").format(python=sys.executable))
        path.chmod(0o755)
    record = tmp_path / "agent_calls.json"
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_AGENT_RECORD", str(record))
    return record
