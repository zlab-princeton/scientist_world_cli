# sciworld

<p align="center">
  <a href="https://ai-scientist.world">
    <img alt="Website" src="https://img.shields.io/badge/%F0%9F%8C%90_Website-ai--scientist.world-2F5496">
  </a>
  <a href="https://arxiv.org/abs/2611.08478">
    <img alt="Paper abstract" src="https://img.shields.io/badge/arXiv-Abstract-b31b1b?logo=arxiv&logoColor=white">
  </a>
  <a href="https://arxiv.org/pdf/2611.08478">
    <img alt="Paper PDF" src="https://img.shields.io/badge/arXiv-PDF-b31b1b?logo=arxiv&logoColor=white">
  </a>
  <a href="https://github.com/orgs/ScientistWorld/repositories">
    <img alt="Environments" src="https://img.shields.io/badge/GitHub-Environments-181717?logo=github&logoColor=white">
  </a>
  <a href="https://huggingface.co/datasets/zlab-princeton/scientist-world-traces">
    <img alt="Traces dataset" src="https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-Traces_dataset-FFD21E">
  </a>
  <a href="https://github.com/zlab-princeton/scientist_world">
    <img alt="Scientist World repository" src="https://img.shields.io/badge/GitHub-Scientist_World-181717?logo=github&logoColor=white">
  </a>
</p>

`sciworld` (`sw`) is the command line of [Scientist World](https://ai-scientist.world). It runs an AI scientist (Claude Code or Codex) in a research environment built from a published paper, and scores the result against the paper's own method.

## Install

```bash
pip install sciworld
```

Needs Python 3.9+, `git`, `bash` (Linux or macOS) and a logged-in [Claude Code](https://docs.anthropic.com/en/docs/claude-code) or [Codex](https://github.com/openai/codex) CLI. A container runtime is optional.

## Quick start

```bash
sw init --data-dir ~/sciworld-data --backend venv    # or plain `sw init` for a guided setup
sw list                                              # browse the environments
sw research 2502.13759 --mode innovate --agent claude-code
sw status                                            # running, finished, failed?
sw evaluate 2502.13759                               # score the run
```

Step by step: [QUICKSTART.md](QUICKSTART.md).

## Modes

| Mode | The agent gets | The agent must |
|---|---|---|
| `improve` | Problem, evaluator, the paper's method and its scores | Beat the paper's method |
| `innovate` | Problem and evaluator only | Design a method of its own |

The agent develops on a **visible** split. When it declares the work done, its method is frozen and scored on a **held-out** split.

## What a run does

1. **Setup.** Downloads the environment, builds a workspace from it, starts the agent in a detached tmux, screen or background session.
2. **Main phase.** The agent builds `method/`, scores it with `scripts/evaluate_train.sh`, writes `idea.md`, sets `{"done": true}` in `status.json`.
3. **Test phase.** The launcher adds the held-out evaluator and locks `method/`. The agent runs `scripts/evaluate_test.sh` and writes `reflection.md`. Finished when `scoring/scores_test.json` exists.

```
<data_dir>/envs/<arxiv_id>/
    source/                the downloaded environment, never changed
    <agent>-<mode>[-<tag>]/    the workspace: all the agent can see and change
    .sciworld/<run>/       agent.log, trajectory/ (raw events, stats.json), launch.sh, prompt_*.md,
                           launch.env (credentials), run.json, launcher.pid, launcher.exit
```

## The workspace

```
READMEs/        problem.md, evaluation.md, TARGETS.md, CONSTRAINTS.md, DIRECTION.md, EXPERIMENTS.md
                improve only: method.md, overview.md, train_reference.json (the scores to beat)
data/           datasets and data-loading code
baseline/       a reference baseline
eval/           the visible evaluator
scripts/        baseline.sh, download.sh, evaluate_train.sh
environment/    container.def, the recipe of the paper's software stack
method/         improve: the paper's method; innovate: empty
scoring/        scores_train.json, scores_test.json
idea.md, reflection.md, status.json
```

Kept out: the paper, the scoring contract (`scoring/reference.json`), earlier scores, builder logs, the held-out evaluator. Where an evaluator needs names from the contract, the agent writes them itself from `READMEs/`.

## Where the agent runs

| Backend | Isolation | Setup |
|---|---|---|
| `apptainer`, `singularity`, `docker`, `podman` | The agent sees only the workspace, at `/home/user`. Briefing, evaluator and baseline are read-only; in the test phase also `method/` and the visible scores. | `sw build` once (5 to 10 minutes) |
| `venv` | **None.** The agent runs as you. | Nothing |

Set with `sw init` or `sw config container`; override per run with `--env`.

- **Unattended.** Agents run with permission prompts disabled. Use `venv` only where you would allow that.
- **GPUs.** Set `container_args`: `["--nv"]` for Apptainer and Singularity, `["--gpus", "all"]` for Docker.
- **Agent image.** Agent CLIs, Python, `uv`, build tools. The agent installs the paper's software itself, from `environment/container.def`.
- **macOS.** Use `venv` or Docker. A container run needs `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN`, because the Claude Code login is in the Keychain.
- **SELinux.** Docker and Podman run with `--security-opt label=disable`.
- **Scope.** The isolation protects your machine. It does not prove that an agent played fair: scores come from the agent's own evaluator runs.

## Scoring

`sw evaluate <arxiv_id>` prints, per scored cell, the paper's value, the run's value and the relative change. Where coefficients exist it also prints **FNI** (Frontier-Normalized Improvement), the leaderboard number:

```
FNI = 100 * sum over cells of  coefficient * (signed gain over the paper's method)
```

- 0 matches the paper. 100 is the typical gain of a frozen reference cohort of frontier models.
- It is the mean of the visible and the held-out split.
- Coefficients live on the `normalization` branch of the environment's repository. `sw evaluate` prints their date: compare FNI values only under the same coefficients.
- Report the method under the paper's method name, or as the only row that is not a scored method.
- No FNI without that branch, or when fewer than three reference models ran the paper.

## Commands

A paper is an arXiv id (`2502.13759`, `arXiv:2502.13759v2` or an arxiv.org URL). Every command accepts `--org NAME` and `--verbose`.

### `sw init`

One-time setup. A wizard without options; with any option it asks nothing.

| Option | Meaning |
|---|---|
| `--data-dir DIR` | Where environments and runs are stored |
| `--backend NAME` | `venv` (default), `apptainer`, `singularity`, `docker`, `podman` |
| `--session NAME` | `tmux`, `screen` or `nohup` |
| `--compute TEXT` | Tells the agent how to run heavy jobs |
| `--build` | Also build the agent image |
| `--yes` | Ask nothing |
| `--force` | Redo an existing setup |

### `sw config`

| Command | Does |
|---|---|
| `sw config` | Walk through all settings |
| `sw config list` | Show them |
| `sw config path` | Print where they are stored |
| `sw config reset` | Delete them (`--yes`: no confirmation). Data is kept |
| `sw config paths`, `compute`, `container`, `session` | Edit one setting |
| `sw config prompt`, `agent-prompt`, `continue-prompt`, `test-prompt` | Edit one prompt |

### `sw build`

Builds the agent image. Not needed for `venv`.

| Option | Meaning |
|---|---|
| `--runtime NAME` | Build for this runtime |
| `--force` | Rebuild |

### `sw list`

Lists the environments. Cached for a day.

| Option | Meaning |
|---|---|
| `--search TEXT` | Filter by id, title or topic |
| `--min-milestone NAME` | Reproduced at least this far: `core_claim`, `core_claim_plus`, `secondary_claims`, `majority`, `near_complete`, `full` |
| `--limit N` | Show at most `N` |
| `--json` | Print JSON |
| `--refresh` | Ignore the cache |

### `sw select`

Downloads an environment, or opens a browser when no id is given. Optional: `sw research` downloads what it needs.

| Option | Meaning |
|---|---|
| `--path DIR` | Download into `DIR` |
| `--overwrite` | Download again |
| `--refresh` | Ignore the cache |

### `sw info`

Describes a downloaded environment and your runs. `--full` adds method, targets, constraints and evaluation.

### `sw research`

Builds a workspace and starts the agent. Asks for anything you did not pass.

| Option | Meaning |
|---|---|
| `--mode improve\|innovate` | The mode |
| `--agent claude-code\|codex` | The agent CLI |
| `--model NAME` | Model for the agent CLI |
| `--env BACKEND` | `venv`, `container` or a runtime, for this run |
| `--tag TAG` | Suffix of the run directory, for several runs |
| `--path DIR` | Build the workspace at `DIR` (the environment goes to `DIR/../source`) |
| `--overwrite` | Replace an existing workspace, never one with an active run |
| `--loop`, `--max-loops N` | Resume an agent that stops early, up to `N` times (default 10) |
| `--setup-only` | Build the workspace and stop |
| `--no-launch` | Write `launch.sh` without starting it, for a scheduler: `sbatch --wrap "bash <launch.sh>"`. Exits 0 only when the run finished |
| `--refresh` | Ignore the cache |

### `sw status`

Shows your runs: `running`, `test phase`, `finished`, `failed`, `stopped` or `not started`, and which scores exist. `--json` prints everything.

### `sw evaluate`

Scores a run. `sw eval` is the same.

| Option | Meaning |
|---|---|
| `--run NAME` | The run, when the paper has several |
| `--path DIR` | Score the workspace at `DIR` |
| `--mode improve\|innovate` | The mode, for a workspace with no record |

## Settings

In `~/.sciworld/config.json` (`SCIWORLD_HOME` moves the directory).

| Key | Meaning |
|---|---|
| `data_dir` | Root of all data. `SCIWORLD_DATA_DIR` overrides it |
| `container` | `venv` or a container runtime |
| `session` | `tmux`, `screen` or `nohup` |
| `org` | GitHub organization of the environments (default `ScientistWorld`). `--org` or `SCIWORLD_ORG` override it |
| `cache_ttl` | Seconds the listing is cached |
| `bind_paths` | Host paths exposed read-only in the container, as `"host:container"`. A bare `"host"` is an executable, mounted in `/usr/local/bin/` |
| `container_args` | Extra arguments for the runtime, such as `["--nv"]` |
| `passthrough_env` | More environment variables to forward to the agent |

- **Compute.** `sw config compute` adds an instruction to the agent's prompt, such as "Submit GPU jobs with `srun --gres=gpu:1 ...`". In a container, expose such commands with `bind_paths`.
- **Prompts.** Shipped in `sciworld/prompts/`. Override with `sw config prompt`, `agent-prompt`, `continue-prompt`, `test-prompt`.
- **GitHub token.** Only private environments need one. Read from `SCIWORLD_GITHUB_TOKEN`, `GITHUB_TOKEN`, `GH_TOKEN`, `gh auth token`, then `~/.git-credentials`.
- **Agent login.** In a container, the login file (`~/.claude/.credentials.json`, `~/.codex/auth.json`) is copied into the workspace. Or export a key first: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`, `CLAUDE_CODE_OAUTH_TOKEN`; `OPENAI_API_KEY`, `CODEX_API_KEY`, `OPENAI_BASE_URL`. Both stay on disk after the run (the copy, and `.sciworld/<run>/launch.env`): delete them when done.

## Python API

```python
import sciworld

sciworld.init(data_dir="~/sciworld-data", backend="venv")
run = sciworld.research(id="2502.13759", agent="claude-code", mode="innovate", loop=True)
print(run["env_dir"], run["log"])

result = sciworld.evaluate(id="2502.13759", run=run["run_name"])
fni = result["fni"]
print(fni["fni"] if fni["fni"] is not None else fni["reason"])

papers = sciworld.list_environments(search="graph")
```

Same options as the commands. The functions never prompt, and raise `RuntimeError` on failure.

## Development

```bash
pip install -e ".[dev]"
pytest
```

No network, container or agent needed. Changes are in [CHANGELOG.md](CHANGELOG.md).

## License

`sciworld` is released under the [MIT License](LICENSE). That license does not cover the environments: each is built from a published paper, and its text, code, data and model weights stay under the license their authors chose.

## Citation

```bibtex
@article{scientist-world,
  title   = {Scientist World: Grow a Research World for AI-Scientists},
  author  = {Lu, Taiming and Konan, Sachin and Liu, Zhuang},
  year    = {2026},
  journal = {arXiv preprint arXiv:2611.08478},
}
```
