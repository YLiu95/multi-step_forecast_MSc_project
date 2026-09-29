# TensorBoard

On the Athena JupyterHub/VS Code host:

```bash
export ARTIFACT_ROOT="/net/tscratch/people/$(id -un)/experiments/experiment_1.5_asc_gpu_nodes"
"$HOME/.venvs/experiment-1.5-gpu/bin/tensorboard" --logdir "$ARTIFACT_ROOT/runs" --host 127.0.0.1 --port 16006
```

Forward port 16006 privately in VS Code. Do not expose the logs through a public tunnel.

## Experiment 1.5: Athena TP + DP Pilot

One signed seven-session cumulative log-return forecast, with 64 distinct tickers,
256 daily-return observations, non-overlapping patches of eight, learned ticker
identity, and hierarchical temporal-then-cross-ticker attention.

The user approved PyTorch, TP=8 within each eight-A100 node, and data parallelism
across nodes. Model size is selected by bounded capacity profiling, including
optimizer-state memory and the ability to finish checkpoint uploads before the
reservation ends. "Largest" means largest tested feasible candidate, not a proof
of the global maximum or evidence of improved forecasting accuracy.

## Deadline and Publication

- Reservation end: 2026-09-29 16:00 UTC+02:00.
- Latest optimizer update cutoff: 15:15; stop earlier if measured backup time requires it.
- Target for verified final backups: 15:50, leaving ten minutes of contingency.
- Public code: this directory in `YLiu95/multi-step_forecast_MSc_project`.
- Public models and aggregate metrics: `YL95/experiment-1.5-asc-gpu-nodes`.
- Private raw data, sample-level records, and tokens must not be published.
- Retain one best model and one latest resumable checkpoint for this pilot.

This is a time-limited pilot, not a completed full-data training study. Best-model
selection uses a fixed pilot validation subset and is labelled accordingly. The
test split is not used for selection or tuning. Implementation and measured results
are recorded in `selfevo_process.md`; claims must follow completed tests/runs.

## Source and Data

The reference architecture and chronological contract come from Experiment 1.2 at
commit `a8e7a7dd55424b6c4d8de0553f9a1f43539a4b05` in this repository. Its code and
checkpoints are unchanged. The data revision is
`YL95/new_experiment_1-data@bcbbefdbe2313673895eb1a0d354747a9f1624fa`.

Credentials are loaded locally from `$HOME/.env` with owner-only permissions.
They are never placed in Git remotes, job arguments, resolved configuration, logs,
or checkpoints. Installation and launch instructions will be completed alongside
the validated executables.