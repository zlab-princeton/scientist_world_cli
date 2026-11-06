## Environment

- You are running inside an isolated container.
- This workspace is a git repo. Commit your changes frequently.
- Files under `READMEs/`, `eval/`, `data/`, `baseline/`, `scripts/evaluate_train.sh`, and `scripts/baseline.sh` are read-only — do not try to modify them.
- SLURM tools (`sbatch`, `srun`, …), nested container runtimes (`apptainer`, `docker`, `podman`, `singularity`), and privilege escalation (`sudo`, `su`) are blocked.
- Do not invent package names. If a tool referenced in the compute instructions is missing, stop and report it rather than guessing something to `pip install`.
