# Changelog

## 0.3.0

First release from `zlab-princeton/scientist_world_cli`.

### Changed

- **The workspace now matches the one used in the Scientist World evaluation.** `scripts/` keeps only `baseline.sh`, `download.sh` and `evaluate_train.sh`; `paper/` and `checkpoints/` are no longer copied; `method/` always exists (empty in innovate mode); improve mode adds `READMEs/train_reference.json`, the scores to beat.
- **`eval/` keeps the modules the visible evaluator imports.** Earlier versions copied `eval/train/` alone, so evaluators that import a module beside it (`eval/metrics.py`, `eval/split_common.py`) failed on import.
- The mode `scratch` is now called `innovate`, as in the paper. `scratch` is still accepted. Run directories of that mode are named `<agent>-innovate`.
- Prompts reach the agent on standard input instead of the command line.
- Papers are given as a positional argument (`sw research 2502.13759`). `--id` still works.
- `sw research` never deletes an existing workspace unless `--overwrite` is given, and never deletes a directory that is not a workspace. With `--overwrite` the old workspace is removed only after the new one is built, and never while a run is active in it.
- **sciworld's own files moved out of the workspace**, to `.sciworld/<run>/` next to it: the launcher, the transcript (`agent.log`), the trajectory, the prompts and the forwarded credentials. The agent can no longer read its own transcript or tamper with what the launcher writes.
- Starting a run never ends an existing tmux or screen session. A session of the same name means that run is still going, and the launch is refused.
- `sw research --setup-only` no longer depends on the configured backend: the workspace keeps the container's paths unless `--env venv` is given.
- `sw select` replaces a download only after the new one succeeded, and refuses any existing directory that is not this paper's environment.
- The launcher exits with status 0 only when the run finished. An unfinished run used to exit 0.
- In venv mode `TMPDIR` is left as it is on the machine. Under the workspace it was often too long for UNIX sockets.
- No prompt is copied into the settings directory by `sw init` any more, so prompt fixes in a new version reach existing installations. Custom task prompts are stored under new names; ones saved by 0.2 are ignored.
- The Python API never asks questions, raises `RuntimeError` with the reason on failure, and returns values instead of exit codes: `evaluate()` the scores, `select()` the location, `init()` and `config()` the settings. `sciworld.config()` takes settings as keyword arguments.

### Added

- `sw list`, `sw status`, `sw build`, `sw config path`.
- `sw init` flags for scripted setup: `--data-dir`, `--backend`, `--session`, `--compute`, `--build`, `--yes`.
- `sw research --model`, `--overwrite`, `--no-launch`; `sw evaluate --path` and `--mode`.
- `sw status` knows whether a launcher is alive from what the launcher records, whatever started it (tmux, screen, nohup, a batch scheduler), and reports a launcher that ended with an error as `failed`.
- `sw research` checks that the launcher came up and reports a run that dies at once, with the end of its log.
- The `container_args` setting, for runtime options such as `--nv`.
- On hosts where SELinux is enforcing, Docker and Podman containers are started with `--security-opt label=disable`, without which they cannot read the workspace.
- `sw build` says how to move the build directory when an Apptainer build runs out of room in it.
- `--org` and `--verbose` are accepted after the command name too.
- FNI, the benchmark score, in `sw evaluate`, including runs whose method is not named like the paper's.
- When a paper has no FNI, `sw evaluate` says why in plain words: no coefficients are published, or the paper is not in the scored pool.
- `sw evaluate` prints the day the coefficients behind an FNI were computed, and returns it as `coefficients_updated`: FNI values are comparable only when they come from the same coefficients.
- `sw evaluate` reads the coefficients from the repository the environment was downloaded from, instead of searching the organization for the paper.
- A GitHub token is optional: public environments work without one. Tokens are also read from `GITHUB_TOKEN`, `GH_TOKEN` and `gh auth token`.
- API-key authentication: the agent's key variables are forwarded to it through a private file.
- An `OPENAI_API_KEY` also reaches Codex as `CODEX_API_KEY`, the only name `codex exec` reads its key from.
- venv mode rewrites the container path `/home/user` in the environment's scripts, which most evaluators hard-code, and in data files where it refers to a file of the workspace.
- `SCIWORLD_HOME`, `SCIWORLD_DATA_DIR`, `SCIWORLD_ORG`.
- A test suite.

### Fixed

- The held-out evaluator, `method/` and the visible scores are now read-only in the test phase. They were left writable because the mounts were decided before those files existed.
- The whole of `eval/` is read-only during a run, not only `eval/train/`.
- A failing agent no longer ends the launcher silently: the exit status is reported and written to the log, `--loop` restarts a session that never started, and three failures in a row stop the run.
- The launcher never follows a symbolic link out of the workspace, neither as a mount source nor when it writes. A link planted by an agent could expose a directory of the host inside the container, or make the launcher overwrite a file of the user.
- A symbolic link in an environment that points outside it is no longer followed when the workspace is built.
- The test phase no longer runs when the held-out evaluator cannot be installed, and no longer deletes the visible evaluator first.
- The launcher ran its own Python helpers from inside the workspace, where Python looks for modules first. A `json.py` or a `sciworld/` directory left there by the agent was imported and run on the host, outside the container. The helpers now run from the run directory.
- For an instant after a launcher ended, it could be seen as gone with no exit status, so `sw research` and `sw status` could misreport a run that had just finished. The launcher now records its exit status before it removes its pid file, and the state is read in that order.
- A run could be reported as unfinished although both phases had completed. The launcher used to set `status.json` back to not done before the test phase and waited for the agent to set it again, which an agent resuming its own session does not reliably do. `status.json` is now left alone, and the test phase is complete when the held-out evaluator has written `scoring/scores_test.json`. Test scores that exist before that evaluator is installed are removed.
- Prompt text can contain any quotes and parentheses. It used to pass through a shell construct that bash 3.2 (macOS) parses differently.
- A request to GitHub that fails or is rate-limited is reported as such, not as "no coefficients published". An expired token no longer blocks public environments.
- `sw evaluate` took a metric's direction from the sign of its coefficient. It now uses `higher_is_better`, like the benchmark.
- `sw evaluate` compared the first row that had a metric, which could be a baseline. It now compares the paper's method with the run's method.
- The GitHub token no longer appears in the clone URL.
- The listing no longer stops at 1000 environments.
- Agent logins copied into a workspace are excluded from its git repository.
- `sciworld.init()` raised `AttributeError`.

### Removed

- `sw create` (use `sw research --setup-only --path DIR`).
- Launching `gemini` and `kimi`: the launcher never supported them. They remain valid labels for `--setup-only` through the Python API.
