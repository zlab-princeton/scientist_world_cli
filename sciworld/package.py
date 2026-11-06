"""Turn a cloned environment repository into a scientist's workspace.

Only an explicit allowlist is copied; everything else in the repository
(the builder's notes, logs, the paper, scores, the held-out evaluator) stays
out of the workspace. The resulting layout is::

    READMEs/        problem.md, evaluation.md, TARGETS.md, CONSTRAINTS.md,
                    DIRECTION.md, EXPERIMENTS.md
                    improve only: method.md, overview.md, train_reference.json
    data/           datasets and data-loading code
    baseline/       reference baseline
    eval/           the visible evaluator: eval/train/ and the modules it imports
    scripts/        baseline.sh, download.sh, evaluate_train.sh
    environment/    container.def, the recipe of the paper's software stack
    method/         improve: the paper's method; innovate: empty
    scoring/        empty; the evaluator writes scores_train.json here
    idea.md, reflection.md, status.json     filled in by the scientist

The held-out evaluator (``eval/test/``, ``scripts/evaluate_test.sh``) is never
part of this workspace. The launcher adds it only after the method is frozen.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
from typing import Optional

from sciworld.fsutil import copy_tree, is_within, make_owner_writable
from sciworld.modes import canonical_mode

# source path -> name under READMEs/
_BRIEFING_COMMON = {
    "briefing/problem.md": "problem.md",
    "briefing/evaluation.md": "evaluation.md",
    "scoring/TARGETS.md": "TARGETS.md",
    "scoring/CONSTRAINTS.md": "CONSTRAINTS.md",
    "scoring/DIRECTION.md": "DIRECTION.md",
    "scoring/EXPERIMENTS.md": "EXPERIMENTS.md",
}
_BRIEFING_IMPROVE = {
    "briefing/method.md": "method.md",
    "briefing/overview.md": "overview.md",
}

# Directories copied whole.
_DIRS_COMMON = ("data", "baseline")
_DIRS_IMPROVE = ("method",)

# The only entry points kept from scripts/. The repository's other scripts
# (run.sh, method.sh, reproduce.sh, evaluate.sh, evaluate_test.sh) drive the
# builder's own reproduction and reference the held-out evaluation.
SCRIPTS_ALLOWED = ("baseline.sh", "download.sh", "evaluate_train.sh")

# Under eval/: always kept, and never kept whatever imports it.
_EVAL_ALWAYS = ("train",)
_EVAL_NEVER = ("test",)

# Files of environment/ that are not part of the recipe: the alternative
# recipe names (one is chosen below), build products, and builder helpers.
_ENVIRONMENT_SKIP = {
    "container.def", "container.final.def", "setup.sh", ".setup_wrapper.sh",
    "container.sif", "overlay.img", ".build_hashes.json",
}

_IDEA_PLACEHOLDER = (
    "<!-- When finished, replace this comment with one clear, complete "
    "statement of the core idea of your method. One paragraph. -->\n"
)
_REFLECTION_PLACEHOLDER = (
    "<!-- After the test phase runs, replace this comment with a short honest "
    "reflection on what you did, how your method works, why it works (or "
    "where it falls short), and anything else future readers should know. "
    "One paragraph. -->\n"
)


def package_environment(
    repo_dir: Path, output_dir: Path, mode: str, skipped: Optional[list] = None,
) -> Path:
    """Build a workspace at *output_dir* from the repository at *repo_dir*.

    *mode* is ``improve`` or ``innovate``. *output_dir* must not exist.
    Returns *output_dir*.

    Entries that cannot or must not be copied are left out and appended to
    *skipped* as ``(path relative to the repository, reason)``: unreadable
    entries, special files, broken symbolic links, and links that point
    outside the repository (following those would copy files of this machine
    into the workspace).
    """
    mode = canonical_mode(mode)
    repo_dir = Path(repo_dir)
    output_dir = Path(output_dir)
    if skipped is None:
        skipped = []
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"environment source not found: {repo_dir}")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    briefing = dict(_BRIEFING_COMMON)
    dirs = list(_DIRS_COMMON)
    if mode == "improve":
        briefing.update(_BRIEFING_IMPROVE)
        dirs += list(_DIRS_IMPROVE)

    readmes = output_dir / "READMEs"
    readmes.mkdir()
    for rel, name in briefing.items():
        _copy_file(repo_dir, rel, readmes / name, skipped)

    for rel in dirs:
        _copy_dir(repo_dir, rel, output_dir / rel, skipped)

    scripts = output_dir / "scripts"
    scripts.mkdir()
    for name in SCRIPTS_ALLOWED:
        _copy_file(repo_dir, f"scripts/{name}", scripts / name, skipped)

    _package_eval(repo_dir, output_dir, skipped)
    _package_environment_dir(repo_dir, output_dir, skipped)

    # The scientist's own space. scoring/ starts empty: neither the scoring
    # contract (reference.json) nor any earlier scores are handed over.
    (output_dir / "scoring").mkdir()
    (output_dir / "method").mkdir(exist_ok=True)
    (output_dir / "idea.md").write_text(_IDEA_PLACEHOLDER)
    (output_dir / "reflection.md").write_text(_REFLECTION_PLACEHOLDER)
    (output_dir / "status.json").write_text(json.dumps({"done": False}, indent=2) + "\n")

    if mode == "improve":
        write_train_reference(repo_dir, output_dir)
    # Read-only entries in the repository must not make the workspace
    # impossible to edit or to remove.
    make_owner_writable(output_dir)
    return output_dir


def _copy_dir(repo_dir: Path, rel: str, dst: Path, skipped: list) -> bool:
    """Copy the directory ``repo_dir/rel`` to *dst*; ``False`` if absent."""
    src = repo_dir / rel
    if not src.is_dir():
        return False
    if src.is_symlink() and not is_within(src, repo_dir):
        skipped.append((rel, "symbolic link to a location outside the environment"))
        return False
    skipped += [(f"{rel}/{path}", reason) for path, reason in copy_tree(src, dst, root=repo_dir)]
    return True


def _copy_file(repo_dir: Path, rel: str, dst: Path, skipped: list) -> bool:
    """Copy the file ``repo_dir/rel`` to *dst*; ``False`` if absent."""
    src = repo_dir / rel
    if not src.is_file():
        return False
    if src.is_symlink() and not is_within(src, repo_dir):
        skipped.append((rel, "symbolic link to a location outside the environment"))
        return False
    try:
        shutil.copy2(src, dst)
    except OSError as exc:
        skipped.append((rel, f"unreadable ({exc.strerror or exc})"))
        return False
    return True


# ── eval/ ─────────────────────────────────────────────────────────────────


def eval_keep_names(eval_dir: Path) -> set:
    """Names directly under ``eval/`` that belong in the workspace.

    ``eval/train/`` alone is not enough: many environments keep shared scoring
    code beside it (``eval/metrics.py``, ``eval/split_common.py``,
    ``eval/src/``) and import it from the train evaluator. Without that code
    the scientist cannot score its own work. So this keeps ``eval/train/`` plus
    the transitive closure of the local modules it imports. ``eval/test/`` is
    excluded unconditionally.
    """
    eval_dir = Path(eval_dir)
    entries = {p.name: p for p in eval_dir.iterdir()}
    keep = {name for name in _EVAL_ALWAYS if name in entries}
    # eval/__init__.py makes ``from eval.metrics import ...`` work.
    if "__init__.py" in entries:
        keep.add("__init__.py")

    def _imported_names(py: Path) -> set:
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            return set()
        names: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    names.add(parts[0])
                    # ``import eval.metrics`` -> eval/metrics
                    if parts[0] == eval_dir.name and len(parts) > 1:
                        names.add(parts[1])
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    # ``from .. import x`` or ``from ..x import y``
                    for alias in node.names:
                        names.add(alias.name.split(".")[0])
                    if node.module:
                        names.add(node.module.split(".")[0])
                elif node.module:
                    parts = node.module.split(".")
                    names.add(parts[0])
                    # ``from eval.metrics import ...`` -> eval/metrics
                    if parts[0] == eval_dir.name and len(parts) > 1:
                        names.add(parts[1])
                    elif parts[0] == eval_dir.name:
                        # ``from eval import metrics``
                        for alias in node.names:
                            names.add(alias.name.split(".")[0])
        return names

    def _py_files(target: Path) -> list:
        if target.is_dir():
            return sorted(target.rglob("*.py"))
        return [target] if target.suffix == ".py" else []

    queue: list = []
    for name in sorted(keep):
        queue.extend(_py_files(entries[name]))
    seen: set = set()
    while queue:
        py = queue.pop()
        if py in seen:
            continue
        seen.add(py)
        for name in _imported_names(py):
            if name in _EVAL_NEVER or name in keep:
                continue
            target = entries.get(name) or entries.get(f"{name}.py")
            if target is None:
                continue  # standard library, third party, or absent
            if target.name in _EVAL_NEVER:
                continue
            keep.add(target.name)
            queue.extend(_py_files(target))
    return keep


def _package_eval(repo_dir: Path, output_dir: Path, skipped: list) -> None:
    src_eval = repo_dir / "eval"
    if not src_eval.is_dir() or (src_eval.is_symlink() and not is_within(src_eval, repo_dir)):
        return
    dst_eval = output_dir / "eval"
    dst_eval.mkdir()
    for name in sorted(eval_keep_names(src_eval)):
        if (src_eval / name).is_dir():
            _copy_dir(repo_dir, f"eval/{name}", dst_eval / name, skipped)
        else:
            _copy_file(repo_dir, f"eval/{name}", dst_eval / name, skipped)


# ── environment/ ──────────────────────────────────────────────────────────


def _package_environment_dir(repo_dir: Path, output_dir: Path, skipped: list) -> None:
    """Ship the container recipe as ``environment/container.def``.

    ``container.final.def`` is preferred over ``container.def`` when both
    exist. ``setup.sh`` and build products are left behind; other files of
    ``environment/`` are copied unchanged.
    """
    src_env = repo_dir / "environment"
    if not src_env.is_dir() or (src_env.is_symlink() and not is_within(src_env, repo_dir)):
        return
    dst_env = output_dir / "environment"
    dst_env.mkdir()
    final_def = src_env / "container.final.def"
    recipe = "container.final.def" if final_def.is_file() else "container.def"
    _copy_file(repo_dir, f"environment/{recipe}", dst_env / "container.def", skipped)
    for entry in sorted(src_env.iterdir()):
        if entry.name in _ENVIRONMENT_SKIP:
            continue
        if entry.is_dir():
            _copy_dir(repo_dir, f"environment/{entry.name}", dst_env / entry.name, skipped)
        else:
            _copy_file(repo_dir, f"environment/{entry.name}", dst_env / entry.name, skipped)


# ── READMEs/train_reference.json ──────────────────────────────────────────


def write_train_reference(repo_dir: Path, output_dir: Path) -> bool:
    """Write ``READMEs/train_reference.json``: the bar to beat in improve mode.

    These are the scores the starting code in ``method/`` produced on the
    visible split (``scoring/scores_train.json`` of the repository), without
    the baseline methods. Returns ``False`` when the repository has no such
    scores; the workspace is still usable without the file.
    """
    scores_path = repo_dir / "scoring" / "scores_train.json"
    reference_path = repo_dir / "scoring" / "reference.json"
    if not is_within(scores_path, repo_dir):
        return False
    try:
        scores = json.loads(scores_path.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(scores, dict):
        return False
    try:
        reference = json.loads(reference_path.read_text()) if is_within(reference_path, repo_dir) else None
    except (OSError, ValueError):
        reference = None

    def _is_baseline(experiment: str, method: str) -> bool:
        if not isinstance(reference, dict):
            return False
        experiments = reference.get("experiments")
        entry = experiments.get(experiment) if isinstance(experiments, dict) else None
        results = entry.get("results") if isinstance(entry, dict) else None
        spec = results.get(method) if isinstance(results, dict) else None
        return isinstance(spec, dict) and spec.get("type") == "baseline"

    wrapped = isinstance(scores.get("experiments"), dict)
    experiments = scores["experiments"] if wrapped else scores
    kept: dict = {}
    for name, experiment in experiments.items():
        if not isinstance(experiment, dict):
            continue
        nested = isinstance(experiment.get("results"), dict)
        results = experiment["results"] if nested else experiment
        methods = {
            method: metrics for method, metrics in results.items()
            if isinstance(metrics, dict) and not _is_baseline(name, method)
        }
        if methods:
            kept[name] = {"results": methods} if nested else methods
    if not kept:
        return False
    readmes = output_dir / "READMEs"
    readmes.mkdir(exist_ok=True)
    out = {"experiments": kept} if wrapped else kept
    (readmes / "train_reference.json").write_text(json.dumps(out, indent=2) + "\n")
    return True
