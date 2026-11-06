"""Settings of the sciworld CLI.

Everything lives in one directory, ``$SCIWORLD_HOME`` (default ``~/.sciworld``)::

    config.json           settings: data directory, backend, session manager, ...
    instructions/*.md     compute instructions and prompt overrides
    cache/                cached environment listings

Environment variables:

``SCIWORLD_HOME``       configuration directory
``SCIWORLD_DATA_DIR``   overrides ``data_dir`` for the current process
``SCIWORLD_ORG``        overrides ``org`` for the current process
"""

from __future__ import annotations

import contextvars
import json
import os
from pathlib import Path

DEFAULT_CONFIG = {
    # GitHub organization that hosts the environments.
    "org": "ScientistWorld",
    # Root of all data: <data_dir>/envs/<arxiv_id>/{source,<run>} and
    # <data_dir>/containers/.
    "data_dir": "",
    # Where the agent runs: "venv" (on this machine, no container) or a
    # container runtime (apptainer, singularity, docker, podman).
    "container": "",
    # How a run is kept alive: tmux, screen or nohup.
    "session": "",
    # Seconds an environment listing stays cached.
    "cache_ttl": 86400,
    # Extra host paths made visible (read-only) inside the agent container:
    # "host_path:container_path", or just "host_path" for an executable, which
    # is then mounted as /usr/local/bin/<name>.
    "bind_paths": [],
    # Extra arguments for the container runtime, for example ["--nv"] to give
    # an Apptainer container the NVIDIA GPUs of this machine.
    "container_args": [],
    # Names of extra environment variables forwarded to the agent.
    "passthrough_env": [],
}


class ConfigError(RuntimeError):
    """The configuration file cannot be read."""


def config_dir() -> Path:
    override = os.environ.get("SCIWORLD_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".sciworld"


def config_file() -> Path:
    return config_dir() / "config.json"


def instructions_dir() -> Path:
    return config_dir() / "instructions"


def is_initialized() -> bool:
    return config_file().exists()


def load_config(apply_env: bool = True) -> dict:
    """Return the settings, with defaults for missing keys.

    With *apply_env* the ``SCIWORLD_DATA_DIR`` and ``SCIWORLD_ORG`` overrides
    are applied. Code that edits and saves the settings passes ``False`` so an
    override is never written back to disk.
    """
    cfg = dict(DEFAULT_CONFIG)
    path = config_file()
    if path.exists():
        try:
            stored = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot read {path}: {exc}") from exc
        if not isinstance(stored, dict):
            raise ConfigError(f"{path} must contain a JSON object")
        cfg.update(stored)
    if apply_env:
        if os.environ.get("SCIWORLD_DATA_DIR"):
            cfg["data_dir"] = os.environ["SCIWORLD_DATA_DIR"]
        if os.environ.get("SCIWORLD_ORG"):
            cfg["org"] = os.environ["SCIWORLD_ORG"]
    if cfg.get("data_dir"):
        cfg["data_dir"] = str(Path(cfg["data_dir"]).expanduser())
    return cfg


def save_config(cfg: dict) -> None:
    path = config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, indent=2) + "\n")
    os.replace(tmp, path)


def containers_dir(cfg: dict) -> Path:
    """Directory that holds the agent image and other shared container files."""
    data_dir = cfg.get("data_dir") or ""
    base = Path(data_dir) if data_dir else config_dir()
    return base / "containers"


# ── Instructions ──────────────────────────────────────────────────────────
#
# An instruction is a markdown file under instructions/. Names in use:
#   system_prompt                  preamble of every prompt
#   compute, compute_gpu, compute_cpu   how the agent should run heavy jobs
#   mode_improve, mode_innovate    replacements for the bundled task prompts
#   continue, test                 replacements for the follow-up prompts
#
# The Python API sets per-call overrides through this context variable, so a
# caller can supply prompts without touching the files on disk.
_instruction_overrides: contextvars.ContextVar = contextvars.ContextVar(
    "sciworld_instruction_overrides", default={}
)


def load_instruction(name: str) -> str:
    """Return the text of an instruction, or ``''`` when it is not set."""
    overrides = _instruction_overrides.get()
    if name in overrides:
        return overrides[name] or ""
    path = instructions_dir() / f"{name}.md"
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def has_instruction(name: str) -> bool:
    """True if the instruction is set, even to an empty text."""
    if name in _instruction_overrides.get():
        return True
    return (instructions_dir() / f"{name}.md").is_file()


def save_instruction(name: str, content: str) -> None:
    directory = instructions_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.md").write_text(content.strip() + "\n")


def remove_instruction(name: str) -> None:
    path = instructions_dir() / f"{name}.md"
    if path.exists():
        path.unlink()


def list_instructions() -> list:
    directory = instructions_dir()
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.md"))
