"""``sw init``, ``sw config`` and ``sw build``: settings and the agent image."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

from sciworld import style as S
from sciworld.settings import (
    ConfigError, DEFAULT_CONFIG, config_dir, config_file, containers_dir,
    has_instruction, instructions_dir, is_initialized, load_config,
    load_instruction, remove_instruction, save_config, save_instruction,
)
from sciworld.launcher import (
    DOCKER_IMAGE_TAG, SUPPORTED_RUNTIMES, SUPPORTED_SESSIONS, agent_image_ref,
    default_session, image_exists,
)
from sciworld.modes import MODES

# Returned by a wizard step when the user chooses "Exit setup".
_EXIT = -1

BACKENDS = ("venv",) + SUPPORTED_RUNTIMES


# ── sw init ───────────────────────────────────────────────────────────────


def cmd_init(args) -> int:
    """First-time setup: interactive wizard, or flags for scripted use."""
    flags = {
        "data_dir": getattr(args, "data_dir", None),
        "backend": getattr(args, "backend", None),
        "session": getattr(args, "session", None),
        "compute": getattr(args, "compute", None),
    }
    build = bool(getattr(args, "build", False))
    scripted = bool(getattr(args, "yes", False)) or build or any(v is not None for v in flags.values())
    force = bool(getattr(args, "force", False))
    try:
        cfg = load_config(apply_env=False) if is_initialized() else dict(DEFAULT_CONFIG)
    except ConfigError as exc:
        print(f"  {S.error(str(exc))}")
        return 1
    if getattr(args, "org", None):
        cfg["org"] = args.org

    if scripted or not S.is_interactive():
        return _init_scripted(cfg, flags, build=build, args=args)

    S.intro(
        "Scientist World",
        prefix="Welcome to",
        subtitle="Every paper with code is a frozen research trajectory.\n"
        "Scientist World turns papers into executable environments\n"
        "in which an AI scientist can improve on the paper's method\n"
        "or design its own.",
    )
    if is_initialized() and not force:
        print(f"  {S.warn(f'Already set up at {S.BOLD}{config_dir()}{S.RESET}')}")
        print(f"  {S.DIM}Change one setting with {S.RESET}{S.BOLD}sw config <section>{S.RESET}"
              f"{S.DIM}, or redo everything with {S.RESET}{S.BOLD}sw init --force{S.RESET}")
        print()
        return 0

    steps = (
        _prompt_data_dir,
        lambda c: _prompt_compute(),
        _prompt_container,
        _prompt_session,
        lambda c: _prompt_system_prompt(),
    )
    for step in steps:
        if step(cfg) == _EXIT:
            break
    save_config(cfg)
    return 0 if _finish_init(cfg, ask_build=True) else 1


def _init_scripted(cfg: dict, flags: dict, build: bool, args=None) -> int:
    """``sw init`` without questions: every value comes from a flag or a default."""
    def fail(message: str) -> int:
        print(f"  {S.error(message)}")
        if args is not None:
            args._sw_error = S.plain(message)
        return 1

    if flags["data_dir"]:
        cfg["data_dir"] = str(Path(flags["data_dir"]).expanduser().resolve())
    if not cfg.get("data_dir"):
        return fail("A data directory is required: sw init --data-dir DIR")
    if flags["backend"]:
        cfg["container"] = flags["backend"]
    elif not cfg.get("container"):
        cfg["container"] = "venv"
    if flags["session"]:
        cfg["session"] = flags["session"]
    elif not cfg.get("session"):
        cfg["session"] = default_session()
    if flags["compute"]:
        save_instruction("compute", flags["compute"])
        remove_instruction("compute_gpu")
        remove_instruction("compute_cpu")

    backend = cfg["container"]
    if backend != "venv" and shutil.which(backend) is None:
        print(f"  {S.warn(f'{backend} is not on PATH; install it before running sw research.')}")
    if cfg["session"] != "nohup" and shutil.which(cfg["session"]) is None:
        return fail(f"{cfg['session']} is not installed. Choose --session tmux, screen or nohup.")
    save_config(cfg)
    if build and backend == "venv":
        print(f"  {S.DIM}--build: the venv backend needs no image.{S.RESET}")
    if not _finish_init(cfg, ask_build=False, build=build):
        return fail("The settings were saved, but the agent image could not be built. Run sw build to try again.")
    return 0


def _finish_init(cfg: dict, ask_build: bool, build: bool = False) -> bool:
    """Create the data layout, optionally build the image, print a summary.

    Returns ``False`` when the image was to be built and the build failed.
    """
    S.bar()
    S.step_done(f"Settings saved to {S.BOLD}{config_file()}{S.RESET}")
    data_dir = cfg.get("data_dir") or ""
    built = True
    if data_dir:
        (Path(data_dir) / "envs").mkdir(parents=True, exist_ok=True)
        (Path(data_dir) / "containers").mkdir(parents=True, exist_ok=True)
        runtime = cfg.get("container") or ""
        if runtime in SUPPORTED_RUNTIMES and (ask_build or build):
            built = build_agent_image(containers_dir(cfg), runtime, ask=ask_build)
    _print_summary(cfg)
    S.bar()
    S.step_end(f"{S.BOLD}Next steps{S.RESET}")
    print(f"     {S.CYAN}${S.RESET} sw list                      {S.DIM}see the available environments{S.RESET}")
    print(f"     {S.CYAN}${S.RESET} sw research <arxiv_id>       {S.DIM}build a workspace and start an agent{S.RESET}")
    print(f"     {S.CYAN}${S.RESET} sw status                    {S.DIM}follow your runs{S.RESET}")
    print()
    return built


# ── Wizard steps ──────────────────────────────────────────────────────────
# Each returns _EXIT when the user leaves the wizard, 0 otherwise. Settings
# go into the cfg dict (saved by the caller); instructions are saved at once.


def _prompt_data_dir(cfg: dict) -> int:
    S.step_start("Data directory")
    S.bar()
    S.note([
        "Where sciworld keeps its data:",
        f"  {S.RESET}<data_dir>/envs/<arxiv_id>/{S.DIM}   downloaded environments and your runs",
        f"  {S.RESET}<data_dir>/containers/{S.DIM}        the agent container image",
        "Environments can be large; pick a disk with room.",
    ])
    S.bar()
    existing = cfg.get("data_dir") or ""
    if existing:
        S.note([f"Current: {S.RESET}{existing}"])
        S.bar()
        choice = S.select("Data directory", ["Keep existing", "Set new path", "Exit setup"], default=0)
        if choice == 0:
            S.bar()
            S.step_done(f"{S.DIM}Kept existing{S.RESET}")
            return 0
        if choice == 2:
            return _exit_setup()
    else:
        choice = S.select("Configure data directory?", ["Set a path", "Skip", "Exit setup"], default=0)
        if choice == 1:
            S.bar()
            S.step_done(f"{S.DIM}Skipped. Set it later with {S.RESET}{S.BOLD}sw config paths{S.RESET}")
            return 0
        if choice == 2:
            return _exit_setup()
    S.bar()
    answer = S.prompt_path("Data directory")
    if not answer.strip():
        S.bar()
        S.step_done(f"{S.DIM}No path given; left unchanged{S.RESET}")
        return 0
    cfg["data_dir"] = str(Path(answer).expanduser().resolve())
    S.bar()
    S.step_done(S.success(f"Set to {S.BOLD}{cfg['data_dir']}{S.RESET}"))
    return 0


def _exit_setup() -> int:
    S.bar()
    S.step_done(f"{S.DIM}Exiting setup{S.RESET}")
    return _EXIT


def _prompt_compute() -> int:
    S.bar()
    S.step_start("Compute")
    S.bar()
    S.note([
        "How the agent should run heavy jobs on your machine. The text is",
        "added to the agent's prompt, so write it as an instruction.",
    ])
    S.bar()
    combined = load_instruction("compute")
    gpu = load_instruction("compute_gpu")
    cpu = load_instruction("compute_cpu")
    if combined or gpu or cpu:
        if combined:
            S.note([f"Current: {S.RESET}{combined}"])
        else:
            S.note([f"Current GPU: {S.RESET}{gpu or '(not set)'}", f"Current CPU: {S.RESET}{cpu or '(not set)'}"])
        S.bar()
        choice = S.select("Compute", [
            "Keep existing",
            "One instruction for GPU and CPU",
            "Separate GPU and CPU instructions",
            "Exit setup",
        ], default=0)
        if choice == 0:
            S.bar()
            S.step_done(f"{S.DIM}Kept existing{S.RESET}")
            return 0
        if choice == 3:
            return _exit_setup()
        separate = choice == 2
    else:
        S.note([
            "Examples:",
            f"  {S.RESET}Run everything directly on this machine; it has one GPU.",
            f"  {S.RESET}Submit GPU jobs with: srun --gres=gpu:1 --mem=32G --time=4:00:00 <cmd>",
        ])
        S.bar()
        choice = S.select("Configure compute?", [
            "One instruction for GPU and CPU",
            "Separate GPU and CPU instructions",
            "Skip",
            "Exit setup",
        ], default=2)
        if choice == 2:
            S.bar()
            S.step_done(f"{S.DIM}Skipped. Set it later with {S.RESET}{S.BOLD}sw config compute{S.RESET}")
            return 0
        if choice == 3:
            return _exit_setup()
        separate = choice == 1

    S.bar()
    if separate:
        save_instruction("compute_gpu", S.prompt("GPU instruction"))
        save_instruction("compute_cpu", S.prompt("CPU instruction"))
        remove_instruction("compute")
    else:
        save_instruction("compute", S.prompt("Compute instruction"))
        remove_instruction("compute_gpu")
        remove_instruction("compute_cpu")
    S.bar()
    S.step_done(S.success("Compute instruction saved"))
    return 0


_INSTALL_HELP = {
    "apptainer": (
        "# Debian/Ubuntu\n"
        "sudo add-apt-repository -y ppa:apptainer/ppa && sudo apt update\n"
        "sudo apt install -y apptainer\n"
        "# Fedora/RHEL\n"
        "sudo dnf install -y apptainer\n"
        "# Docs: https://apptainer.org/docs/admin/main/installation.html"
    ),
    "singularity": "# Docs: https://docs.sylabs.io/guides/latest/admin-guide/installation.html",
    "docker": "# Linux: https://docs.docker.com/engine/install/\n# macOS/Windows: https://docs.docker.com/desktop/",
    "podman": (
        "# Debian/Ubuntu: sudo apt install -y podman\n"
        "# Fedora/RHEL:   sudo dnf install -y podman\n"
        "# macOS:         brew install podman\n"
        "# Docs: https://podman.io/docs/installation"
    ),
}


def _detect_runtimes() -> dict:
    return {name: shutil.which(name) for name in SUPPORTED_RUNTIMES if shutil.which(name)}


def _runtime_works(name: str) -> bool:
    try:
        return subprocess.run([name, "--version"], capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _wait_for_install(name: str) -> bool:
    """Show install instructions; True once the runtime is on PATH."""
    while True:
        S.bar()
        for line in _INSTALL_HELP.get(name, "").split("\n"):
            S.bar(f"{S.DIM}{line}{S.RESET}")
        S.bar()
        if S.select(f"After installing {name}", ["Check again", "Back"]) == 1:
            return False
        if shutil.which(name):
            S.bar()
            S.step_done(S.success(f"{name} is now available"))
            return True
        S.bar()
        S.step_done(S.error(f"{name} still not found in PATH"))


def _prompt_container(cfg: dict) -> int:
    S.bar()
    S.step_start("Where the agent runs")
    S.bar()
    S.note([
        f"{S.RESET}container{S.DIM} (apptainer, singularity, docker, podman): the agent is isolated",
        "from the rest of your machine and the evaluator is write-protected.",
        f"{S.RESET}venv{S.DIM}: no container, nothing to build. The agent runs directly on",
        "this machine with your own permissions and is not sandboxed.",
    ])
    S.bar()
    existing = cfg.get("container") or ""
    if existing:
        S.note([f"Current: {S.RESET}{existing}"])
        S.bar()

    while True:
        found = _detect_runtimes()
        options, values = [], []
        if existing:
            options.append("Keep existing")
            values.append("__keep__")
        for name in SUPPORTED_RUNTIMES:
            if name in found:
                options.append(f"{name}  {S.GREEN}found at{S.RESET} {S.DIM}{found[name]}{S.RESET}")
            else:
                options.append(f"{name}  {S.DIM}not installed{S.RESET}")
            values.append(name)
        options.append(f"venv  {S.DIM}no container, not sandboxed{S.RESET}")
        values.append("venv")
        options += ["Skip", "Exit setup"]
        values += ["__skip__", "__exit__"]

        default = 0
        if not existing:
            default = next((values.index(n) for n in SUPPORTED_RUNTIMES if n in found),
                           values.index("venv"))
        selected = values[S.select("Backend", options, default=default)]
        if selected == "__keep__":
            S.bar()
            S.step_done(f"{S.DIM}Kept existing ({existing}){S.RESET}")
            return 0
        if selected == "__exit__":
            return _exit_setup()
        if selected == "__skip__":
            S.bar()
            S.step_done(f"{S.DIM}Skipped. Set it later with {S.RESET}{S.BOLD}sw config container{S.RESET}")
            return 0
        if selected == "venv":
            cfg["container"] = "venv"
            S.bar()
            S.step_done(S.success("Using venv: the agent runs on this machine, without a sandbox"))
            return 0
        if selected not in found and not _wait_for_install(selected):
            continue
        S.bar()
        if not _runtime_works(selected):
            S.step_done(S.error(f"{selected} was found but `{selected} --version` failed"))
            continue
        cfg["container"] = selected
        S.step_done(S.success(f"Using {S.BOLD}{selected}{S.RESET}"))
        return 0


_SESSION_NOTES = {
    "tmux": "persistent session you can attach to",
    "screen": "persistent session you can attach to",
    "nohup": "plain background process with a log file",
}


def _prompt_session(cfg: dict) -> int:
    S.bar()
    S.step_start("Session manager")
    S.bar()
    S.note(["What keeps a run alive after you close the terminal."])
    S.bar()
    existing = cfg.get("session") or ""
    if existing:
        S.note([f"Current: {S.RESET}{existing}"])
        S.bar()
    found = {name: shutil.which(name) for name in ("tmux", "screen") if shutil.which(name)}
    found["nohup"] = "always available"

    options, values = [], []
    if existing:
        options.append("Keep existing")
        values.append("__keep__")
    for name in SUPPORTED_SESSIONS:
        state = f"{S.GREEN}available{S.RESET}" if name in found else f"{S.DIM}not installed{S.RESET}"
        options.append(f"{name}  {state}  {S.DIM}{_SESSION_NOTES[name]}{S.RESET}")
        values.append(name)
    options += ["Skip", "Exit setup"]
    values += ["__skip__", "__exit__"]
    default = 0 if existing else values.index(default_session())
    selected = values[S.select("Session manager", options, default=default)]

    if selected == "__keep__":
        S.bar()
        S.step_done(f"{S.DIM}Kept existing ({existing}){S.RESET}")
        return 0
    if selected == "__exit__":
        return _exit_setup()
    if selected == "__skip__":
        S.bar()
        S.step_done(f"{S.DIM}Skipped. Set it later with {S.RESET}{S.BOLD}sw config session{S.RESET}")
        return 0
    S.bar()
    if selected not in found:
        S.step_done(S.error(f"{selected} is not installed; install it and run `sw config session`"))
        return 0
    cfg["session"] = selected
    S.step_done(S.success(f"Using {S.BOLD}{selected}{S.RESET}"))
    return 0


def _default_system_prompt() -> str:
    from sciworld.research_cmd import bundled_prompt
    return bundled_prompt("system_prompt")


def _prompt_system_prompt() -> int:
    S.bar()
    S.step_start("System prompt")
    S.bar()
    S.note([
        "The preamble of every prompt: the agent's role and how autonomous it is.",
        "sciworld ships a default; write your own only if you want another one.",
    ])
    S.bar()
    default = _default_system_prompt()
    custom = has_instruction("system_prompt")
    current = load_instruction("system_prompt") if custom else default
    S.note([f"Current ({'your own' if custom else 'default'}):{S.RESET}"])
    for line in textwrap.wrap(current, width=60):
        S.note([f"  {S.RESET}{line}"])
    S.bar()
    options = ["Keep it", "Write your own"]
    if custom:
        options.append("Reset to the default")
    options.append("Exit setup")
    choice = options[S.select("System prompt", options, default=0)]
    S.bar()
    if choice == "Keep it":
        S.step_done(f"{S.DIM}Kept{S.RESET}")
        return 0
    if choice == "Exit setup":
        return _exit_setup()
    if choice == "Reset to the default":
        remove_instruction("system_prompt")
        S.step_done(S.success("System prompt reset to the default"))
        return 0
    text = S.prompt_ghost("System prompt", current)
    S.bar()
    if text.strip() == default.strip():
        # The default is not stored, so later improvements to it apply.
        remove_instruction("system_prompt")
        S.step_done(S.success("Using the default system prompt"))
    else:
        save_instruction("system_prompt", text)
        S.step_done(S.success("System prompt saved"))
    return 0


def _edit_prompt(title: str, name: str, default_text: str, note: list) -> int:
    """Edit or reset one prompt override stored as instruction *name*."""
    S.bar()
    S.step_start(title)
    S.bar()
    S.note(note)
    S.bar()
    custom = load_instruction(name)
    S.bar(f"{S.DIM}({'customized' if custom else 'default'}){S.RESET}")
    S.bar()
    choice = S.select("What to do?", ["Edit", "Reset to the default", "Skip"], default=2)
    S.bar()
    if choice == 0:
        save_instruction(name, S.prompt_ghost(title, custom or default_text))
        S.bar()
        S.step_done(S.success(f"{title} saved"))
    elif choice == 1:
        remove_instruction(name)
        S.step_done(S.success(f"{title} reset to the default"))
    else:
        S.step_done(f"{S.DIM}Skipped{S.RESET}")
    return 0


def _prompt_agent_prompt() -> int:
    from sciworld.research_cmd import bundled_prompt
    for mode in MODES:
        _edit_prompt(
            f"Task prompt ({mode})", f"mode_{mode}", bundled_prompt(mode),
            [f"The task given to the agent in {S.RESET}{mode}{S.DIM} mode."],
        )
    return 0


def _prompt_continue_prompt() -> int:
    from sciworld.research_cmd import bundled_prompt
    return _edit_prompt(
        "Continue prompt", "continue", bundled_prompt("continue"),
        [f"Sent each time {S.RESET}sw research --loop{S.DIM} resumes the agent's session."],
    )


def _prompt_test_prompt() -> int:
    from sciworld.research_cmd import bundled_prompt
    return _edit_prompt(
        "Test prompt", "test", bundled_prompt("test"),
        ["Sent once, after the main phase, when the held-out evaluator is added."],
    )


# ── Summary ───────────────────────────────────────────────────────────────


def _summary_rows(cfg: dict) -> list:
    unset = f"{S.DIM}not set{S.RESET}"
    rows = [
        ("org", cfg.get("org") or unset),
        ("data", cfg.get("data_dir") or unset),
        ("backend", cfg.get("container") or unset),
        ("session", cfg.get("session") or unset),
    ]
    combined = load_instruction("compute")
    if combined:
        rows.append(("compute", combined))
    else:
        gpu, cpu = load_instruction("compute_gpu"), load_instruction("compute_cpu")
        if gpu or cpu:
            rows += [("gpu", gpu or unset), ("cpu", cpu or unset)]
        else:
            rows.append(("compute", unset))
    if has_instruction("system_prompt"):
        rows.append(("prompt", load_instruction("system_prompt") or f"{S.DIM}(empty){S.RESET}"))
    else:
        rows.append(("prompt", f"{S.DIM}default{S.RESET}"))
    return rows


def _print_summary(cfg: dict, indent: str = "") -> None:
    width = max(S._term_cols() - 24, 30)
    S.bar()
    for label, value in _summary_rows(cfg):
        lines = textwrap.wrap(value, width=width, break_long_words=False) or [value]
        S.bar(f"{indent}{S.DIM}{label:<8}{S.RESET} {lines[0]}")
        for line in lines[1:]:
            S.bar(f"{indent}{' ' * 9}{line}")


# ── sw config ─────────────────────────────────────────────────────────────

CONFIG_SECTIONS = {
    "paths": _prompt_data_dir,
    "compute": lambda cfg: _prompt_compute(),
    "container": _prompt_container,
    "session": _prompt_session,
    "prompt": lambda cfg: _prompt_system_prompt(),
    "agent-prompt": lambda cfg: _prompt_agent_prompt(),
    "continue-prompt": lambda cfg: _prompt_continue_prompt(),
    "test-prompt": lambda cfg: _prompt_test_prompt(),
}
CONFIG_ACTIONS = ("list", "path", "reset")


def _config_list(cfg: dict) -> int:
    print()
    print(f"  {S.BOLD}Settings{S.RESET}  {S.DIM}{config_file()}{S.RESET}")
    for label, value in _summary_rows(cfg)[:4]:
        print(f"    {S.DIM}{label:<8}{S.RESET} {value}")
    print()
    print(f"  {S.BOLD}Instructions{S.RESET}  {S.DIM}{instructions_dir()}{S.RESET}")
    width = max(S._term_cols() - 18, 30)
    for label, value in _summary_rows(cfg)[4:]:
        lines = textwrap.wrap(value, width=width, break_long_words=False) or [value]
        print(f"    {S.DIM}{label:<8}{S.RESET} {lines[0]}")
        for line in lines[1:]:
            print(f"    {' ' * 9}{line}")
    print()
    print(f"  {S.BOLD}Prompts{S.RESET}")
    for label, name in [(f"task ({m})", f"mode_{m}") for m in MODES] + [("continue", "continue"), ("test", "test")]:
        state = "customized" if load_instruction(name) else f"{S.DIM}default{S.RESET}"
        print(f"    {S.DIM}{label:<16}{S.RESET} {state}")
    print()
    return 0


def _config_reset(assume_yes: bool) -> int:
    targets = [config_file(), instructions_dir(), config_dir() / "cache"]
    if not any(t.exists() for t in targets):
        print(f"\n  {S.DIM}Nothing to reset.{S.RESET}\n")
        return 0
    if not assume_yes:
        if not S.is_interactive():
            print(f"  {S.error('Refusing to reset without confirmation. Pass --yes.')}")
            return 1
        print()
        if S.select(f"Delete the settings, instructions and cache in {config_dir()}?",
                    ["Cancel", "Delete"]) == 0:
            print(f"  {S.DIM}Cancelled.{S.RESET}\n")
            return 0
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    print(f"\n  {S.success(f'Settings cleared in {S.BOLD}{config_dir()}{S.RESET}')}")
    print(f"  {S.DIM}Downloaded environments and runs were not touched. "
          f"Run {S.RESET}{S.BOLD}sw init{S.RESET}{S.DIM} to start again.{S.RESET}\n")
    return 0


def cmd_config(args) -> int:
    """Show or change the settings."""
    section = getattr(args, "section", None)
    if section == "path":
        print(config_dir())
        return 0
    if section == "reset":
        return _config_reset(bool(getattr(args, "yes", False)))
    try:
        cfg = load_config(apply_env=False)
    except ConfigError as exc:
        print(f"  {S.error(str(exc))}")
        return 1
    if section == "list":
        return _config_list(cfg)
    if not is_initialized():
        print(f"  {S.error(f'Not set up yet. Run {S.BOLD}sw init{S.RESET} first.')}\n")
        return 1
    if section and section not in CONFIG_SECTIONS:
        print(f"  {S.error(f'Unknown section: {section}')}")
        print(f"  {S.DIM}Available: {', '.join(CONFIG_ACTIONS + tuple(CONFIG_SECTIONS))}{S.RESET}\n")
        return 1
    if not S.is_interactive():
        print(f"  {S.error('sw config <section> asks questions and needs a terminal.')}")
        print(f"  {S.DIM}For scripts use: sw init --data-dir DIR --backend NAME --session NAME{S.RESET}\n")
        return 1

    S.intro("Scientist World", subtitle="Configuration")
    S.step_start("Current settings")
    _print_summary(cfg)
    S.bar()
    steps = [CONFIG_SECTIONS[section]] if section else list(CONFIG_SECTIONS.values())
    for step in steps:
        if step(cfg) == _EXIT:
            break
    save_config(cfg)
    S.bar()
    S.step_end(f"Saved to {S.BOLD}{config_file()}{S.RESET}")
    print()
    return 0


# ── sw build ──────────────────────────────────────────────────────────────


# What Apptainer and Singularity print when the directory they build in is
# too small, or on a file system that cannot hold a container's files.
_BUILD_DIR_SYMPTOMS = re.compile(r"no space left on device|disk quota exceeded|xattr", re.IGNORECASE)


def _recipe_path(runtime: str) -> Path:
    name = "agent.def" if runtime in ("apptainer", "singularity") else "agent.Dockerfile"
    return Path(__file__).parent / "containers" / name


def build_agent_image(containers: Path, runtime: str, *, ask: bool = False, force: bool = False) -> bool:
    """Build the agent image: ``<containers>/agent.sif`` for apptainer and
    singularity, the tag ``sciworld-agent:latest`` for docker and podman.

    With *ask*, offers to skip. Returns ``True`` when the image is available
    afterwards or the user skipped, ``False`` when the build failed.
    """
    if runtime not in SUPPORTED_RUNTIMES:
        S.bar()
        S.step_done(S.error(f"Unsupported container runtime: {runtime}"))
        return False
    recipe = _recipe_path(runtime)
    image = agent_image_ref(runtime, containers)
    if image_exists(runtime, image) and not force:
        S.bar()
        S.step_done(f"Agent image already built: {S.BOLD}{image}{S.RESET}")
        return True
    if shutil.which(runtime) is None:
        S.bar()
        S.step_done(S.error(f"{runtime} not found in PATH"))
        return False
    if ask:
        S.bar()
        S.note([
            "The agent image holds the agent CLIs (Claude Code, Codex), Python and uv.",
            "Building it takes 5 to 10 minutes and needs network access.",
        ])
        S.bar()
        if S.select("Build the agent image now?", ["Build now", "Later (sw build)"]) == 1:
            S.bar()
            S.step_done(f"{S.DIM}Skipped. Build it later with {S.RESET}{S.BOLD}sw build{S.RESET}")
            return True

    Path(containers).mkdir(parents=True, exist_ok=True)
    if runtime in ("apptainer", "singularity"):
        command = [runtime, "build", "--force", image, str(recipe)]
    else:
        command = [runtime, "build", "--tag", DOCKER_IMAGE_TAG, "--file", str(recipe), str(recipe.parent)]

    S.bar()
    label = f"Building the agent image with {runtime}"
    S.step_start(label)
    started = time.time()
    finished = threading.Event()
    live = S.is_interactive()

    def _tick() -> None:
        while not finished.wait(1.0):
            if live:
                elapsed = int(time.time() - started)
                sys.stdout.write(f"{S.MOVE_UP}{S.CLEAR_LINE}")
                S.step_start(f"{label}  {S.DIM}{elapsed // 60}m{elapsed % 60:02d}s{S.RESET}")
                sys.stdout.flush()

    ticker = threading.Thread(target=_tick, daemon=True)
    ticker.start()
    hint = None
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
        output = (result.stderr or result.stdout).strip()
        failure = None if result.returncode == 0 else output[-600:]
        if failure is not None and runtime in ("apptainer", "singularity") and _BUILD_DIR_SYMPTOMS.search(output):
            variable = f"{runtime.upper()}_TMPDIR"
            hint = [
                f"{runtime} unpacks the image under ${variable} (default: /tmp) before it writes the .sif.",
                f"It needs a few GB there, on a local disk. Set {variable} to such a directory and run sw build again.",
            ]
    except subprocess.TimeoutExpired:
        failure = "timed out after 60 minutes"
    except OSError as exc:
        failure = str(exc)
    finally:
        finished.set()
        ticker.join(timeout=2)
    if live:
        sys.stdout.write(f"{S.MOVE_UP}{S.CLEAR_LINE}")
    if failure is not None:
        S.step_done(S.error(f"Build failed: {failure}"))
        if hint:
            S.note(hint)
        return False
    elapsed = int(time.time() - started)
    S.step_done(S.success(
        f"Built {S.BOLD}{image}{S.RESET}  {S.DIM}({elapsed // 60}m{elapsed % 60:02d}s){S.RESET}"
    ))
    return True


def cmd_build(args) -> int:
    """Build the agent container image for the configured runtime."""
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"  {S.error(str(exc))}")
        return 1
    runtime = getattr(args, "runtime", None) or cfg.get("container") or ""
    S.intro("Scientist World", subtitle="Agent container image")
    if runtime == "venv":
        S.step_done("The venv backend needs no image.")
        print()
        return 0
    if runtime not in SUPPORTED_RUNTIMES:
        S.step_done(S.error(
            f"No container runtime configured. Pass --runtime {{{','.join(SUPPORTED_RUNTIMES)}}} "
            f"or run {S.BOLD}sw config container{S.RESET}."
        ))
        print()
        return 1
    ok = build_agent_image(containers_dir(cfg), runtime, force=bool(getattr(args, "force", False)))
    print()
    return 0 if ok else 1
