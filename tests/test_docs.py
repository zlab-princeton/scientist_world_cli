"""The documentation must describe the program that is shipped."""

import re
import shlex
from pathlib import Path

import pytest

from sciworld import cli
from sciworld.settings import DEFAULT_CONFIG

ROOT = Path(__file__).parent.parent
DOCS = [ROOT / "README.md", ROOT / "QUICKSTART.md"]


def documented_commands():
    """Every `sw ...` line inside a fenced code block of the documentation."""
    found = []
    for doc in DOCS:
        in_block = False
        for line in doc.read_text().splitlines():
            if line.startswith("```"):
                in_block = not in_block
                continue
            line = line.strip()
            if in_block and line.startswith("sw "):
                found.append((doc.name, line.split("#")[0].strip()))
    return found


@pytest.mark.parametrize("doc, command", documented_commands())
def test_documented_commands_parse(doc, command):
    argv = shlex.split(command)[1:]
    if "--version" in argv:
        return
    try:
        cli.build_parser().parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"{doc}: `{command}` is rejected by the parser (exit {exc.code})")


def test_the_documentation_has_examples():
    assert len(documented_commands()) >= 12


def command_sections():
    """The text of each ``### `sw <name>` `` section of the README."""
    readme = (ROOT / "README.md").read_text()
    sections = {}
    for match in re.finditer(r"^### `sw (\w+)`\n(.*?)(?=^##|\Z)", readme, flags=re.M | re.S):
        sections[match.group(1)] = match.group(2)
    return sections


def test_every_command_has_its_own_section_with_all_of_its_options():
    """An option counts as documented only where its command is described:
    `--path` means different things for `select`, `research` and `evaluate`."""
    sections = command_sections()
    readme = (ROOT / "README.md").read_text()
    parser = cli.build_parser()
    subparsers = next(a for a in parser._actions if a.dest == "command")
    everywhere = {"--org", "--verbose"}   # described once, for all commands
    for flag in everywhere:
        assert f"`{flag}" in readme
    for name, sub in subparsers.choices.items():
        if name == "eval":
            continue
        assert name in sections, f"command {name} has no section in the README"
        for action in sub._actions:
            for flag in action.option_strings:
                if not flag.startswith("--") or flag == "--help" or action.help == "==SUPPRESS==":
                    continue
                if flag in everywhere:
                    continue
                assert f"`{flag}" in sections[name], f"`sw {name} {flag}` is not described in its section"
    for section in ("list", "path", "reset", "paths", "compute", "container", "session",
                    "prompt", "agent-prompt", "continue-prompt", "test-prompt"):
        assert section in sections["config"], section


def test_the_python_api_example_names_keys_that_exist(configured):
    """Every key the README reads from a result must be one the API returns."""
    import sciworld
    readme = (ROOT / "README.md").read_text()
    run = sciworld.research(id="2401.00001", agent="claude-code", mode="innovate", setup_only=True)
    for key in re.findall(r'run\["(\w+)"\]', readme):
        assert key in run, key
    from sciworld.status_cmd import run_status
    row = run_status("2401.00001", run["env_dir"])
    for state in re.findall(r"`(running|test phase|finished|failed|stopped|not started)`", readme):
        assert state  # the states named in the README are the ones the code can produce
    assert row["state"] in readme


def test_every_setting_is_documented():
    readme = (ROOT / "README.md").read_text()
    for key in DEFAULT_CONFIG:
        assert f"`{key}`" in readme, f"setting {key} is not in the README"


def test_workspace_layout_in_the_readme_matches_the_packager(env_source, tmp_path):
    from sciworld.package import package_environment
    readme = (ROOT / "README.md").read_text()
    workspace = package_environment(env_source, tmp_path / "ws", "improve")
    for entry in sorted(p.name for p in workspace.iterdir()):
        name = entry + "/" if (workspace / entry).is_dir() else entry
        assert name in readme, f"{name} is in the workspace but not in the README"
    for name in sorted(p.name for p in (workspace / "READMEs").iterdir()):
        assert name in readme
    for name in sorted(p.name for p in (workspace / "scripts").iterdir()):
        assert name in readme


def test_changelog_names_the_current_version():
    import sciworld
    assert f"## {sciworld.__version__}" in (ROOT / "CHANGELOG.md").read_text()


def test_no_path_of_a_particular_machine():
    """Nothing shipped points into somebody's own directories. The one home
    directory that may be named is the container's, ``/home/user``."""
    pattern = re.compile(r"/scratch/|/Users/[A-Za-z]|/home/(?!user)[A-Za-z]")
    for path in list((ROOT / "sciworld").rglob("*")) + [p for p in ROOT.glob("*") if p.is_file()]:
        if path.is_file() and path.suffix in {".py", ".md", ".toml", ".def", ".Dockerfile", ""}:
            hits = [l for l in path.read_text(errors="replace").splitlines() if pattern.search(l)]
            assert not hits, f"{path.relative_to(ROOT)}: {hits[:3]}"
