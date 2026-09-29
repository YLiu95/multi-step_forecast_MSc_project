# TensorBoard

On the Athena JupyterHub/VS Code host:

```bash
export ARTIFACT_ROOT="/net/tscratch/people/$(id -un)/experiments/experiment_1.5_asc_gpu_nodes"
"$HOME/.venvs/experiment-1.5-gpu/bin/tensorboard" --logdir "$ARTIFACT_ROOT/runs" --host 127.0.0.1 --port 16007
```

Forward port 16007 privately in VS Code. Port 16006 was already occupied and was left untouched. Do not expose the logs through a public tunnel.

## Experiment 1.5: Athena TP + DP Pilot

One signed seven-session cumulative log-return forecast, with 64 distinct tickers,
256 daily-return observations, non-overlapping patches of eight, learned ticker
identity, and hierarchical temporal-then-cross-ticker attention.

The user approved PyTorch, TP=8 within each eight-A100 node, and data parallelism
across nodes. Model size is selected by bounded capacity profiling, including
optimizer-state memory and the ability to finish checkpoint uploads before the
reservation ends. "Largest" means largest tested feasible candidate, not a proof
of the global maximum or evidence of improved forecasting accuracy.

For the complete chronology, decisions, failed attempts, validation scope, and
backup evidence, see [WORK_PROCESS_REPORT.md](WORK_PROCESS_REPORT.md).

## Measured Pilot Outcome

Job `3210061` ran the 7,444,254,721-parameter model with **TP=8, DP=12 on 96 A100s**.
It completed **21 optimizer updates and 6,720 unique training examples**, about
0.0096% of the 69,858,966 eligible training anchors. This is an operational pilot,
not meaningful convergence evidence.

| Measurement | Result |
| --- | --- |
| Fixed pilot validation subset | 839 anchors, market-stratified with available Mag 7 targets |
| Pilot signed Huber | 5.023262 |
| Zero-return baseline Huber, same subset | 5.004593 |
| Pilot MAE / RMSE | 5.490915 / 8.567953 log-return percentage points |
| Prediction / target standard deviation | 0.012205 / 8.563569 percentage points |
| Best and latest checkpoint | Both update 21, with different inference/recovery formats |
| Last optimizer update | 2026-09-29 14:59:32 UTC+02 |
| Checkpoints and summary complete | About 15:03 UTC+02 |
| First verified remote artifact set | 104,331,123,677 bytes at 15:17:57 UTC+02 |
| Main job exit | FAILED, exit 1, after complete artifacts were written |

**The model did not beat the zero-return baseline.** Predictions were nearly
constant after this very short run. Do not infer a structural failure or a
capacity benefit from 21 updates; run a smaller matched-data baseline and a much
longer controlled budget before interpreting forecasting quality. No full epoch,
full validation, or test-set evaluation was completed.

Full TP+DP training recorded **37.78 GiB peak reserved memory on rank 0**, tighter
than the one-node profile suggested. This is reserved, not measured live allocated
memory, and it is not an all-rank maximum. Reduce microbatch size and measure all
rank peaks before a longer run; the one-node headroom estimate did not carry over
unchanged to DDP.

The original distributed-exit error is retained in the operational report; its
exact cause was not conclusively established. Explicit DP/TP/world teardown and
persistent per-rank error logs were added. A subsequent tiny TP=8/DP=12 diagnostic
including DDP-gradient equivalence completed cleanly in job `3210183`. The full
7.44B run was not restarted after the training cutoff.

## Deadline and Publication

- Reservation end: 2026-09-29 16:00 UTC+02:00.
- Pilot optimizer update cutoff: 15:00 UTC+02, moved earlier after measuring upload throughput.
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
or checkpoints. The source commit recorded in each checkpoint identifies the
implementation used for that training state.

## Verified Setup and Selected Pilot

Run `source ./env.sh` from this directory. This keeps the shared Python/OpenSSL
libraries while filtering cluster CUDA/NCCL libraries that conflict with the
PyTorch CUDA 12.4 wheel stack. Install `requirements.txt` in the private environment;
never modify the shared tutorial environment.

`python -m pytest -q tests` validates labels, session masks, sampling, artifact
allowlisting, and full optimizer checkpoint restoration. The distributed checker
`ascgpu.parallel_check` passed on CPU TP=2 and GPU TP=8/DP=2. A three-update real-data
TP=8/DP=2 smoke run also completed and serialized recovery state.

The selected pilot has **7,444,254,721 parameters**, width 4096, 20 temporal blocks,
16 cross-ticker blocks, and microbatch 16 per data replica. The 16.63B candidate ran
but exceeded the 85% reserved-memory target and could not be backed up within this
reservation. The selected candidate used 32.90 GB peak reserved memory out of
42.60 GB reported device capacity. Eight-shard upload probing measured about
41.56 MB/s, which motivated the earlier shutdown and constrained model size.

This is the largest tested candidate selected under both memory and backup-time
constraints, not a proof that no intermediate architecture could fit.

## Run and Resume

The launcher runs one `torchrun` per allocated node, with eight GPU processes per
node. Use `--nodes=1-48 --ntasks-per-node=1 --gpus-per-node=8 --cpus-per-task=32
--mem=256G` with `salloc --immediate=60`. Do not fix the total task count to 48.
Export `EXPERIMENT_DIR` to this directory before submitting, then run:

```bash
srun --ntasks-per-node=1 --cpus-per-task=32 bash "$EXPERIMENT_DIR/node_entry.sh" \
	ascgpu.train --root "$ARTIFACT_ROOT" --config "$EXPERIMENT_DIR/configs/pilot.json" \
	--global-batch 320 --microbatch 16 --train-until 2026-09-29T15:00:00+02:00
```

Choose a wall-time ending before reservation expiry. This dated deadline must be
changed for a later approved allocation. `--global-batch 320` counts distinct real
examples across DP replicas, not TP ranks; padding is masked. Add `--resume` with
a completed local checkpoint directory to restore optimizer and sampler state.

Run `python -m ascgpu.artifacts watch --root "$ARTIFACT_ROOT"` on the shared-storage
JupyterHub host to back up live aggregate logs and publish best/latest after the
training completion marker. Final publishing compares remote sizes and Git/LFS
digests. Do not delete local checkpoints if verification fails.