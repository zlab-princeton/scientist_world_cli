import json
import os
import shutil

import pytest

from sciworld.package import SCRIPTS_ALLOWED, eval_keep_names, package_environment, write_train_reference


def tree(root):
    return sorted(
        str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
    )


def test_innovate_workspace_contract(env_source, tmp_path):
    out = package_environment(env_source, tmp_path / "ws", "innovate")
    assert tree(out) == [
        "READMEs/CONSTRAINTS.md", "READMEs/DIRECTION.md", "READMEs/EXPERIMENTS.md",
        "READMEs/TARGETS.md", "READMEs/evaluation.md", "READMEs/problem.md",
        "baseline/majority.py",
        "data/loader.py", "data/numbers.json",
        "environment/container.def",
        "eval/__init__.py", "eval/common.py", "eval/metrics.py",
        "eval/train/__init__.py", "eval/train/evaluate.py", "eval/train/labels.json",
        "idea.md", "reflection.md",
        "scripts/baseline.sh", "scripts/download.sh", "scripts/evaluate_train.sh",
        "status.json",
    ]
    assert (out / "method").is_dir() and not any((out / "method").iterdir())
    assert (out / "scoring").is_dir() and not any((out / "scoring").iterdir())
    assert json.loads((out / "status.json").read_text()) == {"done": False}


def test_improve_adds_the_method_and_its_reference_scores(env_source, tmp_path):
    out = package_environment(env_source, tmp_path / "ws", "improve")
    extra = set(tree(out)) - set(tree(package_environment(env_source, tmp_path / "ws2", "innovate")))
    assert extra == {"READMEs/method.md", "READMEs/overview.md", "READMEs/train_reference.json",
                     "method/predict.py"}
    reference = json.loads((out / "READMEs" / "train_reference.json").read_text())
    # The baseline row is left out; the paper's method is the bar to beat.
    assert reference == {"experiments": {"main": {"results": {"paper_method": {"accuracy": 0.5}}}}}


@pytest.mark.parametrize("mode", ["improve", "innovate"])
def test_nothing_about_the_held_out_split_or_the_builder_leaks(env_source, tmp_path, mode):
    out = package_environment(env_source, tmp_path / "ws", mode)
    names = tree(out)
    assert not any(n.startswith(("eval/test", "paper/", "checkpoints/", "jobs/")) for n in names)
    assert sorted(os.listdir(out / "scripts")) == sorted(SCRIPTS_ALLOWED)
    for forbidden in ("scoring/reference.json", "scoring/scores_train.json", "scoring/scores_test.json",
                      "scoring/scores.json", "PROGRESS.md", "README.md",
                      "environment/setup.sh", "environment/container.sif", "eval/unused.py"):
        assert not (out / forbidden).exists(), forbidden
    assert "evaluate_test" not in " ".join(names)


def test_scratch_is_accepted_as_innovate(env_source, tmp_path):
    out = package_environment(env_source, tmp_path / "ws", "scratch")
    assert not (out / "READMEs" / "method.md").exists()
    with pytest.raises(ValueError):
        package_environment(env_source, tmp_path / "ws3", "reproduce")


def test_refuses_an_existing_output_and_a_missing_source(env_source, tmp_path):
    (tmp_path / "ws").mkdir()
    with pytest.raises(FileExistsError):
        package_environment(env_source, tmp_path / "ws", "improve")
    with pytest.raises(FileNotFoundError):
        package_environment(tmp_path / "nope", tmp_path / "ws4", "improve")


def test_final_recipe_is_preferred_and_extra_files_are_kept(env_source, tmp_path):
    (env_source / "environment" / "container.final.def").write_text("FINAL\n")
    (env_source / "environment" / "requirements.txt").write_text("numpy\n")
    (env_source / "environment" / "assets").mkdir()
    (env_source / "environment" / "assets" / "a.txt").write_text("a\n")
    out = package_environment(env_source, tmp_path / "ws", "innovate")
    assert (out / "environment" / "container.def").read_text() == "FINAL\n"
    assert sorted(os.listdir(out / "environment")) == ["assets", "container.def", "requirements.txt"]


def test_symbolic_links_inside_the_repository_are_followed(env_source, tmp_path):
    """Relative links are what git stores. They must arrive as real files,
    whatever the current directory of the process is."""
    (env_source / "data" / "v2").mkdir()
    (env_source / "data" / "v2" / "table.csv").write_text("a,b\n")
    (env_source / "data" / "table.csv").symlink_to("v2/table.csv")            # relative, to a file
    (env_source / "data" / "latest").symlink_to("v2")                         # relative, to a directory
    (env_source / "data" / "linked.json").symlink_to(env_source / "data" / "numbers.json")  # absolute, inside
    (env_source / "method" / "default.py").symlink_to("predict.py")
    (env_source / "briefing" / "method.md").unlink()
    (env_source / "briefing" / "real_method.md").write_text("# Method\n")
    (env_source / "briefing" / "method.md").symlink_to("real_method.md")
    skipped = []
    out = package_environment(env_source, tmp_path / "ws", "improve", skipped=skipped)
    assert skipped == []
    assert (out / "data" / "table.csv").read_text() == "a,b\n"
    assert (out / "data" / "latest" / "table.csv").read_text() == "a,b\n"
    assert (out / "data" / "linked.json").read_text() == (env_source / "data" / "numbers.json").read_text()
    assert (out / "method" / "default.py").read_text() == (env_source / "method" / "predict.py").read_text()
    assert (out / "READMEs" / "method.md").read_text() == "# Method\n"
    for path in out.rglob("*"):
        assert not path.is_symlink(), path


def test_links_that_leave_the_repository_are_never_followed(env_source, tmp_path):
    """An environment is third-party content: a link to /etc/passwd or to the
    user's home must not copy that file into a workspace."""
    host_file = tmp_path / "host_secret.txt"
    host_file.write_text("PRIVATE\n")
    host_dir = tmp_path / "host_dir"
    host_dir.mkdir()
    (host_dir / "key").write_text("PRIVATE\n")
    (env_source / "data" / "leak.txt").symlink_to(host_file)
    (env_source / "data" / "leakdir").symlink_to(host_dir)
    (env_source / "data" / "dangling").symlink_to(tmp_path / "gone")
    (env_source / "data" / "loop").symlink_to(".")
    (env_source / "scripts" / "download.sh").unlink()
    (env_source / "scripts" / "download.sh").symlink_to(host_file)
    shutil.rmtree(env_source / "baseline")
    (env_source / "baseline").symlink_to(host_dir)
    (env_source / "scoring" / "scores_train.json").unlink()
    (env_source / "scoring" / "scores_train.json").symlink_to(host_file)
    skipped = []
    out = package_environment(env_source, tmp_path / "ws", "improve", skipped=skipped)
    for path in out.rglob("*"):
        if path.is_file() and not path.is_symlink():
            assert "PRIVATE" not in path.read_text(errors="replace"), path
    assert not (out / "data" / "leak.txt").exists() and not (out / "data" / "leakdir").exists()
    assert not (out / "data" / "dangling").exists() and not (out / "baseline").exists()
    assert not (out / "scripts" / "download.sh").exists()
    assert (out / "data" / "numbers.json").exists()  # the rest is there
    reasons = dict(skipped)
    assert "outside the environment" in reasons["data/leak.txt"]
    assert "outside the environment" in reasons["data/leakdir"]
    assert "outside the environment" in reasons["baseline"]
    assert "outside the environment" in reasons["scripts/download.sh"]
    assert reasons["data/dangling"] == "broken symbolic link"
    assert reasons["data/loop"] == "symbolic link loop"


def test_unreadable_and_special_entries_are_left_out_not_fatal(env_source, tmp_path):
    """Seen in a real environment: a root file system committed under
    environment/ with directories nobody can read and device-like entries."""
    locked = env_source / "environment" / "rootfs" / "var" / "run" / "udisks2"
    locked.mkdir(parents=True)
    (locked / "x").write_text("x\n")
    (env_source / "environment" / "rootfs" / "etc").mkdir()
    (env_source / "environment" / "rootfs" / "etc" / "os-release").write_text("ID=test\n")
    os.mkfifo(env_source / "environment" / "rootfs" / "fifo")
    os.chmod(locked, 0)
    skipped = []
    try:
        out = package_environment(env_source, tmp_path / "ws", "innovate", skipped=skipped)
    finally:
        os.chmod(locked, 0o700)
    assert (out / "environment" / "rootfs" / "etc" / "os-release").read_text() == "ID=test\n"
    assert not (out / "environment" / "rootfs" / "fifo").exists()
    reasons = dict(skipped)
    assert reasons["environment/rootfs/fifo"] == "not a regular file"
    if os.geteuid() != 0:  # root can read everything
        assert "unreadable" in reasons["environment/rootfs/var/run/udisks2"]


def test_a_minimal_repository_still_packages(tmp_path):
    repo = tmp_path / "repo"
    (repo / "briefing").mkdir(parents=True)
    (repo / "briefing" / "problem.md").write_text("p\n")
    out = package_environment(repo, tmp_path / "ws", "improve")
    assert tree(out) == ["READMEs/problem.md", "idea.md", "reflection.md", "status.json"]
    assert (out / "method").is_dir() and (out / "scoring").is_dir() and (out / "scripts").is_dir()


# ── eval/ closure ──

def _eval_dir(tmp_path, files):
    root = tmp_path / "eval"
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def test_keeps_a_sibling_module_the_train_evaluator_imports(tmp_path):
    root = _eval_dir(tmp_path, {
        "train/evaluate.py": "import sys\nsys.path.insert(0, '..')\nimport split_common\n",
        "split_common.py": "X = 1\n", "other.py": "Y = 2\n", "test/evaluate.py": "import split_common\n",
    })
    assert eval_keep_names(root) == {"train", "split_common.py"}


def test_follows_imports_transitively(tmp_path):
    root = _eval_dir(tmp_path, {
        "train/evaluate.py": "from split_common import f\n",
        "split_common.py": "import metrics\n", "metrics.py": "import os\n", "unused.py": "",
    })
    assert eval_keep_names(root) == {"train", "split_common.py", "metrics.py"}


@pytest.mark.parametrize("statement", [
    "from eval.metrics import accuracy", "from eval import metrics", "import eval.metrics",
    "from ..metrics import accuracy", "from .. import metrics",
])
def test_package_style_imports(tmp_path, statement):
    root = _eval_dir(tmp_path, {
        "__init__.py": "", "train/evaluate.py": statement + "\n", "metrics.py": "", "junk.py": "",
    })
    assert eval_keep_names(root) == {"train", "__init__.py", "metrics.py"}


def test_a_package_directory_is_kept_whole(tmp_path):
    root = _eval_dir(tmp_path, {
        "train/evaluate.py": "from src.scoring import score\n",
        "src/__init__.py": "", "src/scoring.py": "import helpers\n", "helpers.py": "",
    })
    assert eval_keep_names(root) == {"train", "src", "helpers.py"}


def test_the_test_slice_is_never_kept(tmp_path):
    root = _eval_dir(tmp_path, {
        "train/evaluate.py": "import test\nfrom test import labels\nfrom eval.test import evaluate\n",
        "test/__init__.py": "", "test/labels.py": "",
    })
    assert eval_keep_names(root) == {"train"}


def test_unparseable_code_is_tolerated(tmp_path):
    root = _eval_dir(tmp_path, {"train/evaluate.py": "def broken(:\n", "metrics.py": ""})
    assert eval_keep_names(root) == {"train"}


# ── train_reference.json ──

def test_reference_keeps_the_layout_of_the_source_scores(tmp_path):
    repo, out = tmp_path / "repo", tmp_path / "out"
    (repo / "scoring").mkdir(parents=True)
    out.mkdir()
    flat = {"exp": {"ours": {"acc": 1}, "base": {"acc": 0}}}
    (repo / "scoring" / "scores_train.json").write_text(json.dumps(flat))
    (repo / "scoring" / "reference.json").write_text(json.dumps(
        {"experiments": {"exp": {"results": {"base": {"type": "baseline"}}}}}))
    assert write_train_reference(repo, out) is True
    assert json.loads((out / "READMEs" / "train_reference.json").read_text()) == {"exp": {"ours": {"acc": 1}}}


def test_reference_without_a_contract_keeps_every_method(tmp_path):
    repo, out = tmp_path / "repo", tmp_path / "out"
    (repo / "scoring").mkdir(parents=True)
    out.mkdir()
    scores = {"experiments": {"exp": {"results": {"a": {"m": 1}, "b": {"m": 2}}}}}
    (repo / "scoring" / "scores_train.json").write_text(json.dumps(scores))
    assert write_train_reference(repo, out) is True
    assert json.loads((out / "READMEs" / "train_reference.json").read_text()) == scores


@pytest.mark.parametrize("content", [None, "not json", "[]", json.dumps({"experiments": {"e": {"results": {}}}})])
def test_reference_is_skipped_when_there_is_nothing_to_report(tmp_path, content):
    repo, out = tmp_path / "repo", tmp_path / "out"
    (repo / "scoring").mkdir(parents=True)
    out.mkdir()
    if content is not None:
        (repo / "scoring" / "scores_train.json").write_text(content)
    assert write_train_reference(repo, out) is False
    assert not (out / "READMEs" / "train_reference.json").exists()
