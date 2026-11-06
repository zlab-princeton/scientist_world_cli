"""``sw research``: build a workspace for a paper and start an agent in it."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from sciworld import __version__
from sciworld import rundir
from sciworld import style as S
from sciworld.settings import (
    ConfigError, containers_dir, has_instruction, load_config, load_instruction,
)
from sciworld.fsutil import remove_tree, sweep_stale_temp
from sciworld.github import DEFAULT_ORG, GitHubError, github_token
from sciworld.modes import MODES, canonical_mode
from sciworld.package import package_environment
from sciworld.papers import (
    check_source, clone_environment, fetch_papers, lookup_paper, normalize_arxiv_id,
)
from sciworld.widgets import dice_select, paginated_list, tabbed_search


# ── Agents ────────────────────────────────────────────────────────────────

_AGENTS = {
    "claude-code": {
        "label": "Claude Code",
        "description": "Anthropic's CLI agent",
        "binary": "claude",
        # Variables forwarded to the agent when they are set on the host.
        "env": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
                "CLAUDE_CODE_OAUTH_TOKEN"),
    },
    "codex": {
        "label": "Codex",
        "description": "OpenAI's CLI agent",
        "binary": "codex",
        "env": ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL"),
    },
}

# Agents ``sw research`` can launch. Other names are accepted with
# ``--setup-only``, where the agent only labels the run directory.
AGENT_NAMES = tuple(_AGENTS)


def agent_label(agent: str) -> str:
    return _AGENTS.get(agent, {}).get("label", agent)


# ── Prompts ───────────────────────────────────────────────────────────────

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def bundled_prompt(name: str) -> str:
    """Text of a prompt shipped with the package (``prompts/<name>.md``).

    A missing file means a broken installation; starting an agent without
    its task would be worse than failing.
    """
    path = _PROMPTS_DIR / f"{name}.md"
    try:
        return path.read_text().strip()
    except OSError as exc:
        raise RuntimeError(
            f"the sciworld installation is incomplete: cannot read {path} ({exc})"
        ) from exc


def mode_prompt(mode: str) -> str:
    """The task prompt of *mode*: the user's override, else the bundled one."""
    mode = canonical_mode(mode)
    return load_instruction(f"mode_{mode}") or bundled_prompt(mode)


def continue_prompt() -> str:
    return load_instruction("continue") or bundled_prompt("continue")


def test_prompt() -> str:
    return load_instruction("test") or bundled_prompt("test")


def system_prompt() -> str:
    """The preamble of every prompt: the user's own text if set, else the
    bundled one."""
    if has_instruction("system_prompt"):
        return load_instruction("system_prompt")
    return bundled_prompt("system_prompt")


def compute_instructions() -> str:
    """The user's instructions on how to run heavy jobs, if any."""
    combined = load_instruction("compute")
    if combined:
        return combined
    gpu = load_instruction("compute_gpu")
    cpu = load_instruction("compute_cpu")
    lines = []
    if gpu:
        lines.append(f"GPU: {gpu}")
    if cpu:
        lines.append(f"CPU: {cpu}")
    return "\n".join(lines)


# ── Backend ───────────────────────────────────────────────────────────────

_RUNTIMES = ("apptainer", "singularity", "docker", "podman")


def resolve_backend(cfg: dict, env_flag: Optional[str]) -> tuple:
    """Decide where the agent runs.

    Returns ``(backend, error)``. *backend* is ``"venv"`` or a container
    runtime name; it is ``None`` when nothing usable is configured, in which
    case *error* says what to do.
    """
    flag = (env_flag or "").strip().lower()
    configured = (cfg.get("container") or "").strip().lower()
    if flag == "venv" or (not flag and configured == "venv"):
        return "venv", None
    if flag in _RUNTIMES:
        return flag, None
    if flag and flag != "container":
        return None, f"unknown backend {env_flag!r}"
    if configured in _RUNTIMES:
        return configured, None
    if flag == "container":
        for name in _RUNTIMES:
            if shutil.which(name):
                return name, None
        return None, "no container runtime found (apptainer, singularity, docker, podman)"
    return None, (
        f"No execution backend is configured. Run {S.BOLD}sw init{S.RESET}, or pass "
        f"{S.BOLD}--env venv{S.RESET} to run on this machine without a container."
    )


# ── Workspace helpers ─────────────────────────────────────────────────────


def is_workspace(path: Path) -> bool:
    """True if *path* is a workspace built by sciworld."""
    return (path / "status.json").is_file() and (path / "READMEs").is_dir()


def has_test_slice(source_dir: Path) -> bool:
    return (
        (source_dir / "eval" / "test").is_dir()
        and (source_dir / "scripts" / "evaluate_test.sh").is_file()
    )


def _fail(message: str, args=None) -> int:
    """Print *message* as the reason the command stops; return exit status 1.

    The plain text is also left on *args* for the Python API to raise.
    """
    S.bar()
    S.step_done(S.error(message))
    print()
    if args is not None:
        args._sw_error = S.plain(message)
    return 1


def _clean_tag(tag: str) -> str:
    """Reduce a run tag to letters, digits, ``.``, ``_`` and ``-``."""
    return "".join(c if c.isalnum() or c in "._" else "-" for c in tag.strip()).strip("-")


def _active_run(env_dir: Path, session: str, name: str) -> Optional[str]:
    """Why the workspace at *env_dir* must not be replaced right now, or
    ``None`` if no run is active in it."""
    from sciworld import launcher as L

    state = rundir.launcher_state(env_dir)
    record = rundir.read_record(env_dir)
    # How the run was started, if sw research started it; a run without a
    # record (sciworld 0.2) used the configured session manager.
    manager = record.get("session_manager") or (None if record else session)
    session_id = record.get("session") or name
    in_session = manager in ("tmux", "screen") and L.session_exists(manager, session_id)

    def stop() -> str:
        if in_session:
            return L.stop_command(manager, session_id)
        return L.stop_command("nohup", session_id, state["pgid"])

    if state["alive"]:
        return (
            f"A run is still active in {env_dir} (launcher pid {state['pid']}). "
            f"Stop it first: {S.BOLD}{stop()}{S.RESET}"
        )
    if state["pid"] is not None and state["alive"] is None:
        return (
            f"A launcher started on {state['host']} may still be running in {env_dir}. "
            f"Stop it there; if it is already gone, delete "
            f"{rundir.run_dir(env_dir) / 'launcher.pid'} and try again."
        )
    if in_session:
        return (
            f"The {manager} session '{session_id}' of this workspace still exists. "
            f"Stop it first: {S.BOLD}{stop()}{S.RESET}"
        )
    return None


def _select_interactively(papers: list) -> Optional[dict]:
    """Paper picker. Returns ``None`` when the user exits."""
    while True:
        S.bar()
        choice = S.select("Select a paper", [
            "Random (dice roll)",
            "Search (by ID, time, or topic)",
            "List all (paginated)",
            "Exit",
        ])
        S.bar()
        if choice == 3:
            return None
        picker = (dice_select, tabbed_search, paginated_list)[choice]
        selected = picker(papers)
        if selected is not None:
            return selected


# ── Command ───────────────────────────────────────────────────────────────


def cmd_research(args) -> int:
    """Build a workspace and, unless told otherwise, launch the agent."""
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
    mode = getattr(args, "mode", None)
    agent = getattr(args, "agent", None)
    model = getattr(args, "model", None)
    tag = getattr(args, "tag", None)
    env_flag = getattr(args, "env", None)
    setup_only = bool(getattr(args, "setup_only", False))
    no_launch = bool(getattr(args, "no_launch", False) or getattr(args, "launcher", False))
    overwrite = bool(getattr(args, "overwrite", False))
    refresh = bool(getattr(args, "refresh", False))
    loop = bool(getattr(args, "loop", False))
    max_loops = int(getattr(args, "max_loops", 10) or 10)
    interactive = S.is_interactive()
    data_dir = cfg.get("data_dir") or ""

    S.intro("Scientist World", subtitle="Research environment setup")

    if mode:
        try:
            mode = canonical_mode(mode)
        except ValueError as exc:
            return fail(str(exc))
    if paper_id:
        paper_id = normalize_arxiv_id(paper_id)
    if not dest_path and not data_dir:
        return fail(
            f"Data directory not set. Run {S.BOLD}sw init{S.RESET} or pass {S.BOLD}--path{S.RESET}."
        )

    # ── Backend ──
    # With --setup-only nothing runs here, so no runtime is required, and the
    # result must not depend on this machine's settings: the workspace keeps
    # the container's paths unless --env venv asks for this machine's.
    if setup_only:
        flag = (env_flag or "").strip().lower()
        backend = "venv" if flag == "venv" else None
    else:
        backend, error = resolve_backend(cfg, env_flag)
        if backend is None:
            return fail(error)

    # ── Paper ──
    def _source_dir_for(arxiv_id: str) -> Path:
        if dest_path:
            return Path(dest_path).expanduser().resolve().parent / "source"
        return Path(data_dir) / "envs" / arxiv_id / "source"

    selected: Optional[dict] = None
    if paper_id:
        existing = _source_dir_for(paper_id)
        if os.path.lexists(str(existing)):
            # Already downloaded: no need to ask GitHub anything, but it must
            # be this paper's environment.
            problem = check_source(existing, paper_id)
            if problem:
                hint = (
                    f" With {S.BOLD}--path{S.RESET} the environment is kept in the directory "
                    f"{S.BOLD}source{S.RESET} next to the workspace, so each paper needs its own "
                    "parent directory."
                ) if dest_path else ""
                return fail(problem + hint)
            selected = {"arxiv_id": paper_id, "clone_url": "", "title": ""}
        else:
            try:
                selected = lookup_paper(paper_id, org)
            except GitHubError as exc:
                return fail(str(exc))
            if selected is None:
                hint = "" if github_token() else (
                    f" {S.DIM}Without a GitHub token only public environments are visible.{S.RESET}"
                )
                return fail(f"No environment found for {S.BOLD}{paper_id}{S.RESET} in {org}.{hint}")
    else:
        if not interactive:
            return fail("No paper given. Pass an arXiv id, for example: sw research 2502.13759")
        papers = fetch_papers(org, refresh=refresh)
        if papers is None:
            print()
            args._sw_error = f"could not list the environments of {org}"
            return 1
        if not papers:
            return fail("No environments found.")
        selected = _select_interactively(papers)
        if selected is None:
            S.step_done(f"{S.DIM}Cancelled.{S.RESET}")
            print()
            return 0
        problem = None
        if os.path.lexists(str(_source_dir_for(selected["arxiv_id"]))):
            problem = check_source(_source_dir_for(selected["arxiv_id"]), selected["arxiv_id"])
        if problem:
            return fail(problem)
    arxiv_id = selected["arxiv_id"]
    S.bar()
    S.step_done(f"Paper: {S.BOLD}{arxiv_id}{S.RESET}  {S.DIM}{selected.get('title', '')}{S.RESET}")

    # ── Mode ──
    if not mode:
        if not interactive:
            return fail(f"No mode given. Pass --mode {' or --mode '.join(MODES)}.")
        S.bar()
        S.note([
            f"{S.RESET}{S.BOLD}improve{S.RESET}{S.DIM}   start from the paper's method and try to beat it",
            f"{S.RESET}{S.BOLD}innovate{S.RESET}{S.DIM}  only the problem is given; design your own method",
        ])
        S.bar()
        mode = MODES[S.select("Research mode", list(MODES))]
    mode_color = S.BRIGHT_MAGENTA if mode == "improve" else S.BRIGHT_CYAN
    S.bar()
    S.step_done(f"Mode: {mode_color}{mode}{S.RESET}")

    # ── Agent ──
    if not agent:
        if not interactive:
            return fail(f"No agent given. Pass --agent {' or --agent '.join(AGENT_NAMES)}.")
        S.bar()
        options = [f"{a['label']}  {S.DIM}{a['description']}{S.RESET}" for a in _AGENTS.values()]
        agent = AGENT_NAMES[S.select("Agent", options)]
    if not setup_only and agent not in _AGENTS:
        return fail(
            f"sw research cannot launch {S.BOLD}{agent}{S.RESET}. Supported agents: "
            f"{', '.join(AGENT_NAMES)}. Use --setup-only to build the workspace "
            "and start another agent yourself."
        )
    S.bar()
    S.step_done(f"Agent: {S.BOLD}{agent_label(agent)}{S.RESET}")

    # ── Tag ──
    if tag is None and not dest_path and interactive:
        S.bar()
        S.note([
            f"Label to distinguish runs (e.g. {S.RESET}v2{S.DIM}, {S.RESET}ablation{S.DIM})",
            f"Type a tag, or {S.RESET}esc{S.DIM} to skip",
        ])
        S.bar()
        tag = S.prompt_tag("Run tag")
    if tag and tag.strip():
        cleaned = _clean_tag(tag)
        if not cleaned:
            return fail(
                f"The tag {tag!r} has no usable characters. Use letters, digits, '.', '_' or '-'."
            )
        tag = cleaned
        S.bar()
        S.step_done(f"Tag: {S.BOLD}{tag}{S.RESET}")
    else:
        tag = None

    # ── Destination ──
    # <data_dir>/envs/<arxiv_id>/source/            the downloaded environment
    # <data_dir>/envs/<arxiv_id>/<run_name>/        the workspace
    # <data_dir>/envs/<arxiv_id>/.sciworld/<run>/   launcher, log, trajectory
    run_name = f"{agent}-{mode}" + (f"-{tag}" if tag else "")
    if dest_path:
        env_dir = Path(dest_path).expanduser().resolve()
    else:
        env_dir = Path(data_dir) / "envs" / arxiv_id / run_name
    source_dir = _source_dir_for(arxiv_id)
    if env_dir == source_dir:
        return fail(f"{env_dir} is where the environment itself is kept. Choose another --path.")

    from sciworld import launcher as L
    from sciworld.session import session_name

    session = (cfg.get("session") or "").strip() or L.default_session()
    name = session_name(arxiv_id, env_dir.name, location=str(env_dir) if dest_path else "")

    # ── Checks that can fail, before anything is created or replaced ──
    if backend == "venv":
        from sciworld.localize import is_safe_workspace_path
        if not is_safe_workspace_path(env_dir):
            return fail(
                f"venv mode cannot use the path {env_dir}: it contains spaces or shell "
                "characters. Choose another data directory or --path, or use a container."
            )
    elif not setup_only and any(c in str(env_dir) for c in ":,"):
        return fail(
            f"The path {env_dir} contains ':' or ',', which container runtimes read as "
            "separators. Choose another data directory or --path."
        )
    image_ref = ""
    if not setup_only:
        if backend == "venv":
            # The agent CLI runs on this machine, so it must be here.
            if shutil.which(_AGENTS[agent]["binary"]) is None:
                return fail(
                    f"{S.BOLD}{_AGENTS[agent]['binary']}{S.RESET} not found in PATH. "
                    f"Install {_AGENTS[agent]['label']} first."
                )
        else:
            if shutil.which(backend) is None:
                return fail(f"Container runtime {S.BOLD}{backend}{S.RESET} not found in PATH.")
            image_ref = L.agent_image_ref(backend, containers_dir(cfg))
            if not L.image_exists(backend, image_ref):
                return fail(
                    f"The agent image for {S.BOLD}{backend}{S.RESET} is not built ({image_ref}). "
                    f"Run {S.BOLD}sw build{S.RESET}."
                )
        if not no_launch:
            if session not in L.SUPPORTED_SESSIONS:
                return fail(
                    f"Unknown session manager {session!r}. Set it with {S.BOLD}sw config session{S.RESET}."
                )
            if session != "nohup" and shutil.which(session) is None:
                return fail(
                    f"{session} is not installed. Choose another session manager with "
                    f"{S.BOLD}sw config session{S.RESET}."
                )

    replace_existing = False
    if os.path.lexists(str(env_dir)):
        if env_dir.is_dir() and not env_dir.is_symlink() and not any(env_dir.iterdir()):
            env_dir.rmdir()
        elif not is_workspace(env_dir):
            return fail(
                f"{env_dir} exists and is not a sciworld workspace. Choose another "
                f"{S.BOLD}--path{S.RESET} or {S.BOLD}--tag{S.RESET}."
            )
        else:
            active = _active_run(env_dir, session, name)
            if active:
                return fail(active)
            if not overwrite and interactive:
                S.bar()
                S.step_start(f"A workspace already exists at {S.BOLD}{env_dir}{S.RESET}")
                S.bar()
                choice = S.select("What to do?", ["Cancel", "Overwrite (delete it and start over)"])
                if choice == 0:
                    S.bar()
                    S.step_done(f"{S.DIM}Cancelled.{S.RESET}")
                    print()
                    return 0
                overwrite = True
            if not overwrite:
                return fail(
                    f"A workspace already exists at {env_dir}. Pass {S.BOLD}--overwrite{S.RESET} "
                    f"to replace it, or {S.BOLD}--tag{S.RESET} to create another run."
                )
            replace_existing = True
    else:
        # No workspace, but a launcher of an earlier one may still be alive.
        state = rundir.launcher_state(env_dir)
        if state["alive"] or (state["pid"] is not None and state["alive"] is None):
            return fail(_active_run(env_dir, session, name))
        if not setup_only and not no_launch and L.session_exists(session, name):
            return fail(
                f"A {session} session named '{name}' already exists and belongs to another run. "
                f"Use {S.BOLD}--tag{S.RESET} to give this run its own name."
            )

    # ── Clone and package ──
    # The new workspace is built next to its final place and swapped in only
    # when it is complete: a failure leaves an existing workspace untouched.
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    sweep_stale_temp(env_dir.parent)
    tmp_env = env_dir.parent / f".{env_dir.name}.tmp-{os.getpid()}"
    old_env = env_dir.parent / f".{env_dir.name}.old-{os.getpid()}"
    skipped: list = []
    try:
        if source_dir.is_dir():
            S.bar()
            S.step_done(f"Using the environment at {S.BOLD}{source_dir}{S.RESET}")
        else:
            clone_url = selected.get("clone_url") or ""
            if not clone_url:
                return fail("No clone URL available for this paper.")
            S.bar()
            S.step_start("Downloading the environment")
            clone_environment(clone_url, source_dir, arxiv_id=arxiv_id)
            S.step_done(S.success("Downloaded"))
            problem = check_source(source_dir, arxiv_id)
            if problem:
                return fail(problem)

        S.bar()
        S.step_start(f"Packaging the {mode_color}{mode}{S.RESET} workspace")
        remove_tree(tmp_env)
        package_environment(source_dir, tmp_env, mode=mode, skipped=skipped)
        if backend == "venv":
            from sciworld.localize import localize_workspace
            localize_workspace(tmp_env, target=env_dir)
        if replace_existing:
            remove_tree(old_env)
            os.replace(env_dir, old_env)
            try:
                os.replace(tmp_env, env_dir)
            except OSError:
                os.replace(old_env, env_dir)  # put the previous workspace back
                raise
        else:
            os.replace(tmp_env, env_dir)
        # Records of an earlier run in this place describe another workspace.
        remove_tree(rundir.run_dir(env_dir))
        S.step_done(S.success("Packaged"))
    except (GitHubError, OSError, ValueError, shutil.Error) as exc:
        return fail(str(exc))
    finally:
        for leftover in (tmp_env, old_env):
            try:
                remove_tree(leftover)
            except OSError:
                pass
    if skipped:
        S.bar()
        S.note([f"{len(skipped)} entries of the environment were left out:"] + [
            f"  {path}  ({reason})" for path, reason in skipped[:5]
        ] + (["  ..."] if len(skipped) > 5 else []))

    S.bar()
    if L.init_git_repo(env_dir):
        S.step_done(S.success("Git repository initialized"))
    else:
        S.step_done(f"{S.DIM}git not available: the workspace is not a repository{S.RESET}")

    S.bar()
    S.step_done(f"Workspace ready at {S.BOLD}{env_dir}{S.RESET}")
    S.bar()
    protected = "" if backend == "venv" else " (read-only in the container)"
    S.note([
        f"  {S.RESET}{S.BOLD}READMEs/{S.RESET}{S.DIM}   the problem, targets and constraints{protected}",
        f"  {S.RESET}{S.BOLD}eval/{S.RESET}{S.DIM}      the visible evaluator{protected}",
        f"  {S.RESET}{S.BOLD}data/{S.RESET}{S.DIM}      datasets",
        f"  {S.RESET}{S.BOLD}baseline/{S.RESET}{S.DIM}  a reference baseline{protected}",
        f"  {S.RESET}{S.BOLD}method/{S.RESET}{S.DIM}    where the method lives",
        f"  {S.RESET}{S.BOLD}scoring/{S.RESET}{S.DIM}   where the evaluator writes the scores",
    ])
    if setup_only and backend != "venv":
        S.note([
            "Paths in the scripts are the container's (/home/user). For a workspace",
            f"that runs directly on this machine, build it with {S.RESET}--env venv{S.DIM}.",
        ])
    S.step_end("")

    in_data_dir = not dest_path
    score_hint = (
        f"sw evaluate {arxiv_id} --run {env_dir.name}" if in_data_dir
        else f"sw evaluate {arxiv_id} --path {env_dir}"
    )
    result = {
        "arxiv_id": arxiv_id,
        "run_name": run_name,
        "mode": mode,
        "agent": agent,
        "env_dir": env_dir,
        "source_dir": source_dir,
        "launch_script": None,
        "session": None,
        "log": None,
    }
    args._sw_result = result
    if setup_only:
        print()
        return 0

    # ── Launcher ──
    print()
    S.step_start(f"Preparing {S.BOLD}{agent_label(agent)}{S.RESET}")

    run_directory = rundir.ensure_run_dir(env_dir)
    for stale in ("launcher.pid", "launcher.exit", "launcher.exit.tmp"):
        try:
            (run_directory / stale).unlink()
        except OSError:
            pass
    venv_dir = env_dir / ".venv"
    if backend == "venv":
        if not (venv_dir / "bin" / "python").exists():
            created = subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)], capture_output=True, text=True,
            )
            if created.returncode != 0:
                return fail(f"could not create the virtualenv: {(created.stderr or '').strip()[-300:]}")
            S.bar(f"  {S.DIM}virtualenv created at {venv_dir}{S.RESET}")

    try:
        main_prompt = L.build_agent_prompt_text(
            system_prompt=system_prompt(),
            mode_prompt=mode_prompt(mode),
            compute_instructions=compute_instructions(),
            environment_note=bundled_prompt("environment_venv" if backend == "venv" else "environment"),
        )
        continue_text = continue_prompt() if loop else ""
        test_text = test_prompt()
    except RuntimeError as exc:
        return fail(str(exc))

    if backend != "venv":
        for note in L.copy_agent_credentials(agent, env_dir):
            mark = f"{S.YELLOW}!{S.RESET} " if note.startswith("WARNING") else ""
            S.bar(f"  {mark}{S.DIM}{note}{S.RESET}")

    # Variables forwarded to the agent: the agent's own authentication
    # variables, plus the ones named in the `passthrough_env` setting.
    names = list(_AGENTS[agent]["env"]) + [str(v) for v in (cfg.get("passthrough_env") or [])]
    forwarded = {n: os.environ[n] for n in dict.fromkeys(names) if os.environ.get(n)}
    # `codex exec` takes its key from CODEX_API_KEY and ignores OPENAI_API_KEY.
    if agent == "codex" and "OPENAI_API_KEY" in forwarded and "CODEX_API_KEY" not in forwarded:
        forwarded["CODEX_API_KEY"] = forwarded["OPENAI_API_KEY"]
    prefix = L.container_env_prefix(backend) if backend != "venv" else None
    env_exports = {f"{prefix}{k}" if prefix else k: v for k, v in forwarded.items()}
    runtime_args: list = [str(a) for a in (cfg.get("container_args") or [])]
    if backend in ("docker", "podman"):
        for variable in forwarded:
            runtime_args += ["--env", variable]
    if forwarded:
        S.bar(f"  {S.DIM}forwarding {', '.join(forwarded)}{S.RESET}")

    # Host paths the user chose to expose inside the container (read-only).
    extra_binds: list = []
    if backend != "venv":
        for entry in cfg.get("bind_paths") or []:
            host, _, target = str(entry).partition(":")
            host_path = Path(os.path.expanduser(host))
            if host_path.exists():
                extra_binds.append((host_path, target or f"/usr/local/bin/{host_path.name}"))
            else:
                S.bar(f"  {S.YELLOW}!{S.RESET} {S.DIM}bind_paths: {host_path} does not exist; skipped{S.RESET}")

    def _wrap(agent_cmd: list) -> list:
        if backend == "venv":
            return L.build_local_cmd(env_dir, venv_dir, agent_cmd)
        # The launcher adds the read-only mounts itself, before every
        # invocation, after checking that none leads out of the workspace.
        return L.build_container_cmd(
            runtime=backend, image=image_ref, workspace=env_dir,
            extra_binds=extra_binds, cmd=agent_cmd, protect=False,
            extra_args=runtime_args + [L.RawShellArg('"${SW_RO[@]}"')],
        )

    try:
        main_cmd = _wrap(L.build_agent_cmd(agent, model=model))
        resume_cmd = _wrap(L.build_agent_cmd(agent, resume=True, model=model)) if loop else None
        test_enabled = has_test_slice(source_dir)
        test_cmd = _wrap(L.build_agent_cmd(agent, resume=True, model=model)) if test_enabled else None
    except ValueError as exc:
        return fail(str(exc))

    env_file = L.write_env_file(run_directory / "launch.env", env_exports)
    log_path = rundir.log_path(env_dir)
    launch_script = L.write_launch_script(
        run_directory / "launch.sh", env_dir, main_cmd, log_path,
        resume_cmd=resume_cmd,
        max_loops=max_loops if loop else 1,
        test_phase_cmd=test_cmd,
        test_source_dir=source_dir if test_enabled else None,
        main_prompt_text=main_prompt,
        continue_prompt_text=continue_text,
        test_prompt_text=test_text if test_enabled else "",
        agent_name=agent,
        traj_dir=run_directory / "trajectory",
        env_file=env_file,
        ro_flag=None if backend == "venv" else L.ro_bind_flag(backend),
        localize=backend == "venv",
        done_hint=f"score it with: {score_hint}",
    )
    rundir.write_record(env_dir, {
        "arxiv_id": arxiv_id,
        "run_name": env_dir.name,
        "mode": mode,
        "agent": agent,
        "model": model,
        "tag": tag,
        "backend": backend,
        "workspace": str(env_dir),
        "source": str(source_dir),
        "session_manager": None if no_launch else session,
        "session": None if no_launch else name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sciworld": __version__,
    })
    result["launch_script"] = launch_script
    result["log"] = log_path
    S.step_done(S.success(f"Launcher written to {S.BOLD}{launch_script}{S.RESET}"))

    if no_launch:
        S.bar()
        S.note([
            f"Not started. Run it with: {S.RESET}{S.BOLD}bash {launch_script}{S.RESET}",
            f"Log: {S.RESET}{log_path}",
        ])
        S.step_end("")
        print()
        return 0

    # ── Launch ──
    ok, message = L.launch_in_session(session, name, launch_script, log_path)
    if not ok:
        return fail(f"Launch failed: {message}")
    result["session"] = name
    state, exit_status = L.wait_for_launcher(env_dir, session, name)
    if state == "failed":
        tail = ""
        try:
            tail = "\n".join(S.plain(log_path.read_text(errors="replace")).splitlines()[-8:])
        except OSError:
            pass
        reason = f" with exit status {exit_status}" if exit_status is not None else ""
        return fail(
            f"The launcher stopped right after it started{reason}. Log: {log_path}"
            + (f"\n{tail}" if tail else "")
        )
    S.bar()
    if state == "finished":
        S.step_done(S.success(f"{agent_label(agent)} has already finished"))
        S.bar()
        S.bar(f"  Log:     {log_path}")
    else:
        S.step_done(S.success(f"{agent_label(agent)} is running under {S.BOLD}{session}{S.RESET}"))
        S.bar()
        for line in message.split("\n"):
            S.bar(f"  {line}")
        if in_data_dir:
            S.bar(f"  Status:  sw status {arxiv_id}")
    S.bar(f"  Score:   {score_hint}")
    S.step_end("")
    print()
    return 0
