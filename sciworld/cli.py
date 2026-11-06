"""Command-line entry point: ``sw`` (alias ``sciworld``)."""

from __future__ import annotations

import argparse
import logging
import sys

from sciworld import __version__
from sciworld import style as S
from sciworld.launcher import SUPPORTED_RUNTIMES, SUPPORTED_SESSIONS
from sciworld.milestones import MILESTONE_ORDER
from sciworld.modes import MODE_CHOICES

_AGENT_CHOICES = ("claude-code", "codex")
_BACKEND_CHOICES = ("container", "venv") + SUPPORTED_RUNTIMES

_EPILOG = """\
typical session:
  sw init                              one-time setup
  sw list                              see the available environments
  sw research 2502.13759               build a workspace and start an agent
  sw status                            follow your runs
  sw evaluate 2502.13759               score a finished run

documentation: https://github.com/zlab-princeton/scientist_world_cli
"""


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number")
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def _add_common(parser: argparse.ArgumentParser) -> None:
    """``--org`` and ``-v`` are also accepted after the command name.

    The default is suppressed so that a value given before the command is
    not overwritten when the option is absent after it.
    """
    parser.add_argument("--org", default=argparse.SUPPRESS, metavar="NAME",
                        help="GitHub organization that hosts the environments")
    parser.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
                        help="print debug logging")


def _add_paper(parser: argparse.ArgumentParser, dest: str, required: bool = False) -> None:
    """A paper is given as a positional arXiv id; ``--id`` is the older spelling."""
    parser.add_argument(
        "arxiv_id_pos", nargs="?" if not required else None, metavar="ARXIV_ID",
        help="arXiv id of the paper, e.g. 2502.13759",
    )
    parser.add_argument("--id", dest=dest, metavar="ARXIV_ID", help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sw",
        description="Scientist World: run an AI scientist inside a research environment built from a paper.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"sciworld {__version__}")
    parser.add_argument("--org", default=None, metavar="NAME",
                        help="GitHub organization that hosts the environments (default: ScientistWorld); "
                             "also accepted after the command")
    parser.add_argument("-v", "--verbose", action="store_true", help="print debug logging")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # init
    p = sub.add_parser(
        "init", help="one-time setup",
        description="Set the data directory, where the agent runs, and the session manager. "
                    "Without flags this is an interactive wizard; with flags it asks nothing.",
    )
    p.add_argument("--data-dir", metavar="DIR", help="where environments and runs are stored")
    p.add_argument("--backend", choices=("venv",) + SUPPORTED_RUNTIMES,
                   help="where the agent runs: a container runtime, or venv (no container, not sandboxed)")
    p.add_argument("--session", choices=SUPPORTED_SESSIONS, help="what keeps a run alive")
    p.add_argument("--compute", metavar="TEXT", help="instruction telling the agent how to run heavy jobs")
    p.add_argument("--build", action="store_true", help="also build the agent container image")
    p.add_argument("-y", "--yes", action="store_true", help="ask nothing; use defaults for what is not given")
    p.add_argument("--force", action="store_true", help="run the wizard again on an existing setup")
    _add_common(p)

    # config
    from sciworld.setup_cmd import CONFIG_ACTIONS, CONFIG_SECTIONS
    p = sub.add_parser(
        "config", help="show or change settings",
        description="`sw config list` prints the settings, `sw config path` prints where they are "
                    "stored, `sw config reset` deletes them. Any other section opens its editor.",
    )
    p.add_argument("section", nargs="?", choices=CONFIG_ACTIONS + tuple(CONFIG_SECTIONS),
                   help="what to show or edit (default: edit every section)")
    p.add_argument("-y", "--yes", action="store_true", help="with reset: do not ask for confirmation")
    _add_common(p)

    # build
    p = sub.add_parser(
        "build", help="build the agent container image",
        description="Build the image the agent runs in. Not needed for the venv backend.",
    )
    p.add_argument("--runtime", choices=SUPPORTED_RUNTIMES, help="default: the configured runtime")
    p.add_argument("--force", action="store_true", help="rebuild even if the image exists")
    _add_common(p)

    # list
    p = sub.add_parser(
        "list", help="list the available environments",
        description="List the environments of the organization. The listing is cached for a day.",
    )
    p.add_argument("--search", metavar="TEXT", help="keep environments whose id, title or topics contain TEXT")
    p.add_argument("--min-milestone", choices=MILESTONE_ORDER, metavar="NAME",
                   help=f"keep environments reproduced at least this far ({', '.join(MILESTONE_ORDER)})")
    p.add_argument("--limit", type=_positive_int, metavar="N", help="show at most N environments")
    p.add_argument("--json", action="store_true", help="print JSON")
    p.add_argument("--refresh", action="store_true", help="ignore the cache")
    _add_common(p)

    # select
    p = sub.add_parser(
        "select", help="browse environments and download one",
        description="Download an environment to <data_dir>/envs/<arxiv_id>/source. "
                    "Without an id, opens an interactive browser.",
    )
    _add_paper(p, "paper_id")
    p.add_argument("--path", dest="dest_path", metavar="DIR",
                   help="download into this directory instead (it becomes the environment itself)")
    p.add_argument("--overwrite", action="store_true", help="download again if already present")
    p.add_argument("--refresh", action="store_true", help="ignore the cached listing")
    _add_common(p)

    # info
    p = sub.add_parser(
        "info", help="describe a downloaded environment",
        description="Show the problem, the experiments and your runs of a downloaded environment.",
    )
    _add_paper(p, "arxiv_id")
    p.add_argument("--full", action="store_true", help="also show method, targets, constraints and evaluation")
    _add_common(p)

    # research
    p = sub.add_parser(
        "research", help="build a workspace and start an agent in it",
        description="Download the environment if needed, build a workspace for the chosen mode, "
                    "and start the agent in a detached session. Anything not given on the command "
                    "line is asked interactively.",
    )
    _add_paper(p, "paper_id")
    p.add_argument("--mode", choices=MODE_CHOICES, metavar="{improve,innovate}",
                   help="improve: start from the paper's method. innovate: only the problem is given")
    p.add_argument("--agent", choices=_AGENT_CHOICES, help="the agent CLI to run")
    p.add_argument("--model", metavar="NAME", help="model passed to the agent CLI (default: the CLI's own)")
    p.add_argument("--env", choices=_BACKEND_CHOICES, metavar="BACKEND",
                   help="where the agent runs for this run: venv, container, or a runtime name "
                        "(default: the configured backend)")
    p.add_argument("--tag", metavar="TAG", help="suffix that distinguishes several runs of one agent and mode")
    p.add_argument("--path", dest="dest_path", metavar="DIR",
                   help="build the workspace at this path; the environment is kept next to it, in source/")
    p.add_argument("--overwrite", action="store_true", help="replace an existing workspace")
    p.add_argument("--loop", action="store_true",
                   help="when the agent stops before marking the run done, resume its session")
    p.add_argument("--max-loops", type=int, default=10, metavar="N",
                   help="with --loop: stop after N agent invocations (default: 10)")
    p.add_argument("--setup-only", action="store_true",
                   help="build the workspace and stop (container paths, unless --env venv)")
    p.add_argument("--no-launch", action="store_true",
                   help="also write the launcher script, but do not start it")
    p.add_argument("--refresh", action="store_true", help="ignore the cached listing")
    p.add_argument("--launcher", action="store_true", help=argparse.SUPPRESS)  # old name of --no-launch
    _add_common(p)

    # status
    p = sub.add_parser(
        "status", help="show your runs",
        description="List the runs under the data directory and whether each is running or finished.",
    )
    _add_paper(p, "arxiv_id")
    p.add_argument("--json", action="store_true", help="print JSON")
    _add_common(p)

    # evaluate
    p = sub.add_parser(
        "evaluate", aliases=["eval"], help="score a run against the paper",
        description="Compare a run's scores with the paper's reproduced method, cell by cell, "
                    "and report FNI, the benchmark score.",
    )
    _add_paper(p, "arxiv_id")
    p.add_argument("--run", metavar="NAME", help="run directory name; asked if several exist")
    p.add_argument("--path", metavar="DIR", help="score the workspace at this path instead")
    p.add_argument("--mode", choices=MODE_CHOICES, metavar="{improve,innovate}",
                   help="the mode of the run, when it was not built by `sw research` on this machine")
    _add_common(p)
    return parser


def _merge_paper(args: argparse.Namespace, dest: str) -> None:
    """Fold the positional id into *dest*; the flag wins if both are given."""
    positional = getattr(args, "arxiv_id_pos", None)
    if getattr(args, dest, None) is None:
        setattr(args, dest, positional)


def _main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    command = args.command
    if command is None:
        S.intro("Scientist World", subtitle="Research environments built from papers.")
        parser.print_help()
        print()
        return 0
    if command == "init":
        from sciworld.setup_cmd import cmd_init
        return cmd_init(args)
    if command == "config":
        from sciworld.setup_cmd import cmd_config
        return cmd_config(args)
    if command == "build":
        from sciworld.setup_cmd import cmd_build
        return cmd_build(args)
    if command == "list":
        from sciworld.list_cmd import cmd_list
        return cmd_list(args)
    if command == "select":
        _merge_paper(args, "paper_id")
        from sciworld.select_cmd import cmd_select
        return cmd_select(args)
    if command == "info":
        _merge_paper(args, "arxiv_id")
        from sciworld.info_cmd import cmd_info
        return cmd_info(args)
    if command == "research":
        _merge_paper(args, "paper_id")
        from sciworld.research_cmd import cmd_research
        return cmd_research(args)
    if command == "status":
        _merge_paper(args, "arxiv_id")
        from sciworld.status_cmd import cmd_status
        return cmd_status(args)
    if command in ("evaluate", "eval"):
        _merge_paper(args, "arxiv_id")
        from sciworld.evaluate_cmd import cmd_evaluate
        return cmd_evaluate(args)
    parser.error(f"unknown command {command!r}")
    return 2


def main(argv=None) -> int:
    try:
        return _main(argv)
    except KeyboardInterrupt:
        print(f"\n  {S.DIM}Interrupted.{S.RESET}\n")
        return 130
    except BrokenPipeError:
        # Output was piped into a command that exited early (e.g. `| head`).
        try:
            sys.stdout.close()
        except OSError:
            pass
        return 0


if __name__ == "__main__":
    sys.exit(main())
