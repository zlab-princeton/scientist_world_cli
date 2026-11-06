"""Python API: the ``sw`` commands as functions.

    import sciworld

    run = sciworld.research(id="2502.13759", agent="claude-code", mode="innovate")
    # {"arxiv_id", "run_name", "mode", "agent", "env_dir", "source_dir",
    #  "launch_script", "session", "log"}
    sciworld.evaluate(id="2502.13759", run=run["run_name"])

The functions never ask questions, also when the calling script runs in a
terminal: a missing required argument raises ``TypeError`` and a command
that fails raises ``RuntimeError`` with the reason. They print the same
progress output as the command line.

Prompt arguments (``system_prompt``, ``mode_prompt``, ``continue_prompt``,
``test_prompt``, ``compute``) take a string, or a ``pathlib.Path`` whose
content is read when the function is called. They apply to that call only
and do not change the settings on disk.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from sciworld import settings as _settings
from sciworld import style as _style
from sciworld.modes import canonical_mode


def _prompt_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, Path):
        return value.expanduser().read_text()
    return str(value)


def _require(**kwargs) -> None:
    missing = [name for name, value in kwargs.items() if value is None]
    if missing:
        raise TypeError(f"missing required argument(s): {', '.join(missing)}")


def _run(name: str, command, args, overrides: Optional[dict] = None) -> None:
    """Run a ``cmd_*`` function without questions, with per-call prompt
    overrides in effect; raise ``RuntimeError`` if it fails."""
    token = _settings._instruction_overrides.set(overrides) if overrides else None
    try:
        with _style.non_interactive():
            code = command(args)
    finally:
        if token is not None:
            _settings._instruction_overrides.reset(token)
    if code != 0:
        reason = getattr(args, "_sw_error", None) or f"exit status {code}"
        raise RuntimeError(f"sciworld.{name} failed: {reason}")


def research(
    *,
    id: str,
    agent: str,
    mode: str,
    model: Optional[str] = None,
    tag: Optional[str] = None,
    path: Any = None,
    env: Optional[str] = None,
    loop: bool = False,
    max_loops: int = 10,
    setup_only: bool = False,
    no_launch: bool = False,
    overwrite: bool = False,
    refresh: bool = False,
    org: Optional[str] = None,
    system_prompt: Any = None,
    mode_prompt: Any = None,
    continue_prompt: Any = None,
    test_prompt: Any = None,
    compute: Any = None,
) -> dict:
    """Build a workspace for a paper and, by default, start the agent in it.

    Args:
        id: arXiv id of the paper.
        agent: ``claude-code`` or ``codex``. With ``setup_only`` any name is
            accepted and only labels the run directory.
        mode: ``improve`` or ``innovate`` (``scratch`` is accepted as an
            older name of ``innovate``).
        model: model name handed to the agent CLI.
        tag: suffix of the run directory, to keep several runs apart.
        path: build the workspace here instead of under the data directory.
            The environment source is placed next to it, in ``source/``.
        env: ``venv``, ``container`` or a container runtime name; default is
            the configured backend.
        loop, max_loops: resume the agent's session until it marks the run
            done, at most ``max_loops`` invocations.
        setup_only: build the workspace and return. Its scripts keep the
            container's paths (``/home/user``) unless ``env="venv"``.
        no_launch: also write the launcher script, but do not start it.
        overwrite: replace an existing workspace instead of failing.

    Returns a dict with ``arxiv_id``, ``run_name``, ``mode``, ``agent``,
    ``env_dir``, ``source_dir``, ``launch_script``, ``session`` and ``log``.
    Raises ``RuntimeError`` with the reason when the command fails.
    """
    _require(id=id, agent=agent, mode=mode)
    mode = canonical_mode(mode)
    from sciworld.research_cmd import cmd_research

    args = SimpleNamespace(
        paper_id=id, agent=agent, mode=mode, model=model, tag=tag,
        dest_path=str(path) if path else None, env=env,
        loop=loop, max_loops=max_loops, setup_only=setup_only,
        no_launch=no_launch, overwrite=overwrite, refresh=refresh, org=org,
    )
    overrides: dict = {}
    for name, value in (
        ("system_prompt", system_prompt), (f"mode_{mode}", mode_prompt),
        ("continue", continue_prompt), ("test", test_prompt), ("compute", compute),
    ):
        if value is not None:
            overrides[name] = _prompt_text(value)

    _run("research", cmd_research, args, overrides)
    result = getattr(args, "_sw_result", None)
    if result is None:
        raise RuntimeError("sciworld.research finished without a result")
    return result


def evaluate(*, id: Optional[str] = None, run: Optional[str] = None,
             path: Any = None, mode: Optional[str] = None,
             org: Optional[str] = None) -> dict:
    """Score a run against the paper.

    Give ``id`` (and ``run`` if the paper has several runs), or ``path``, the
    workspace directory; both together name the paper of a workspace that is
    outside the data directory. ``mode`` states the mode of a run that was
    not started by ``sw research`` on this machine. Returns
    ``{"splits": ..., "fni": ..., "mode": ...}`` as described in
    :func:`sciworld.evaluate_cmd.evaluate_run`.
    """
    if id is None and path is None:
        raise TypeError("missing required argument: id or path")
    from sciworld.evaluate_cmd import cmd_evaluate

    args = SimpleNamespace(
        arxiv_id=id, run=run, path=str(path) if path else None, mode=mode, org=org,
    )
    _run("evaluate", cmd_evaluate, args)
    return args._sw_result


def select(*, id: str, path: Any = None, overwrite: bool = False,
           refresh: bool = False, org: Optional[str] = None) -> Path:
    """Download the environment of a paper and return where it is.

    Without ``overwrite`` an environment that is already there is kept.
    """
    _require(id=id)
    from sciworld.select_cmd import cmd_select

    args = SimpleNamespace(
        paper_id=id, dest_path=str(path) if path else None,
        overwrite=overwrite, refresh=refresh, org=org,
    )
    _run("select", cmd_select, args)
    return args._sw_result


def info(*, id: str, full: bool = False) -> None:
    """Print the description of a downloaded environment."""
    _require(id=id)
    from sciworld.info_cmd import cmd_info

    _run("info", cmd_info, SimpleNamespace(arxiv_id=id, full=full))


def list_environments(*, org: Optional[str] = None, refresh: bool = False,
                      min_milestone: str = "", search: str = "") -> list:
    """Return the available environments as a list of dicts.

    Each has ``arxiv_id``, ``title``, ``milestone``, ``topics``, ``repo_name``,
    ``clone_url``, ``html_url``, ``private``. Raises ``RuntimeError`` if the
    listing cannot be fetched.
    """
    from sciworld.github import DEFAULT_ORG
    from sciworld.list_cmd import filter_papers
    from sciworld.papers import fetch_papers

    cfg = _settings.load_config()
    papers = fetch_papers(org or cfg.get("org") or DEFAULT_ORG, refresh=refresh, quiet=True)
    if papers is None:
        raise RuntimeError("could not list the environments (network, rate limit or access)")
    return filter_papers(papers, min_milestone, search)


def config(**changes) -> dict:
    """Return the settings; with keyword arguments, update and save them first.

    ``sciworld.config()`` reads. ``sciworld.config(data_dir="/data/sw",
    container="venv")`` writes. Unknown keys raise ``KeyError``.
    """
    cfg = _settings.load_config(apply_env=not changes)
    if changes:
        unknown = sorted(set(changes) - set(_settings.DEFAULT_CONFIG))
        if unknown:
            raise KeyError(f"unknown setting(s): {', '.join(unknown)}")
        cfg.update(changes)
        _settings.save_config(cfg)
    return dict(cfg)


def init(*, data_dir: Any, backend: str = "venv", session: Optional[str] = None,
         compute: Optional[str] = None, build: bool = False, org: Optional[str] = None) -> dict:
    """Set sciworld up without questions and return the settings.

    ``backend`` is ``venv`` or a container runtime (``apptainer``,
    ``singularity``, ``docker``, ``podman``); ``build`` also builds the agent
    image of a container runtime, and a failed build raises ``RuntimeError``.
    """
    _require(data_dir=data_dir)
    from sciworld.setup_cmd import cmd_init

    args = SimpleNamespace(
        data_dir=str(data_dir), backend=backend, session=session, compute=compute,
        build=build, yes=True, force=False, org=org,
    )
    _run("init", cmd_init, args)
    return config()
