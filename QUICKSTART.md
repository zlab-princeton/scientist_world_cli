# Quick start

Ten minutes from nothing to a running AI scientist. This uses the `venv` backend, which needs no container. Read [Where the agent runs](README.md#where-the-agent-runs) before using it on a machine you care about: in this mode the agent is not sandboxed.

## 1. Install

```bash
pip install sciworld
sw --version
```

You also need an agent CLI that is logged in. Check with `claude --version` or `codex --version`.

## 2. Set up

```bash
sw init --data-dir ~/sciworld-data --backend venv
```

Plain `sw init` asks the same questions interactively, and also offers the container backends.

## 3. Pick an environment

```bash
sw list                      # every environment you can access
sw list --search diffusion   # filter by id, title or topic
sw select 2502.13759         # download one
sw info 2502.13759           # read what it is about
```

Without a GitHub token you see the public environments. To see the ones you were given access to, `export GITHUB_TOKEN=...` first.

## 4. Start a scientist

```bash
sw research 2502.13759 --mode innovate --agent claude-code --loop
```

`innovate` gives the agent only the problem; `improve` also gives it the paper's method to beat. `--loop` resumes the agent if it stops before finishing. The command returns once the agent is running and prints how to watch and how to stop the run:

```bash
sw status                              # running, test phase, finished, failed, stopped
tail -f ~/sciworld-data/envs/2502.13759/.sciworld/claude-code-innovate/agent.log
tmux attach -t '=sw_2502_13759_claude-code-innovate'
```

The agent works in `~/sciworld-data/envs/2502.13759/claude-code-innovate/`. The log, the launcher and the other files sciworld keeps about the run are next to it, in `.sciworld/claude-code-innovate/`.

A run takes as long as the research takes: often hours, and it uses your agent subscription or API credit throughout.

## 5. Score it

```bash
sw evaluate 2502.13759
```

You get the run's value next to the paper's for every metric. For environments that have published normalization coefficients you also get FNI, the benchmark score: 0 matches the paper's method, above 0 improves on it. The example paper is not in the scored pool, so for it `sw evaluate` ends with the per-metric comparison and says why there is no FNI.

## Next

- Isolate the agent: `sw config container`, pick a runtime, then `sw build`.
- Tell the agent how to use your GPUs: `sw config compute`. In a container, also pass them through with the `container_args` setting.
- Run several attempts side by side: `--tag v2`.
- Submit a run to a scheduler: add `--no-launch` and submit the printed `launch.sh`.
- Drive everything from Python: see the [Python API](README.md#python-api).
