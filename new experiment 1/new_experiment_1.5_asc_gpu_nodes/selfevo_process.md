# Experiment Diary

## 2026-09-29: Authorized Pilot

- The user approved TP=8 per node and DP equal to node count, superseding the earlier DP=1 restriction.
- PyTorch 2.6.0 with CUDA 12.4 wheels was selected; shared tutorial software is unchanged.
- Source pinned to `a8e7a7dd55424b6c4d8de0553f9a1f43539a4b05`; data pinned to `bcbbefdbe2313673895eb1a0d354747a9f1624fa`.
- The user explicitly approved public publication of weights and aggregate metrics to the named Hugging Face repository. Raw data and credentials remain private.
- Latest training cutoff is 15:15 UTC+02, backup verification target 15:50, reservation ends 16:00. Measured transfer time may require an earlier stop.
- The pilot keeps the signed seven-session target and chronological split contract. A fixed pilot validation subset selects the pilot best checkpoint; this is not full-validation model selection, and no test metrics will be used.
- Model capacity must be measured including FP32 optimizer state, rematerialized activations, and backup cost. No largest-model or training-performance result is claimed yet.

## Measured Gates and Pilot Decision

- All eight unit tests passed, including full optimizer-state restoration.
- CPU TP=2 forward/gradient equivalence passed; GPU TP=8/DP=2 BF16 equivalence and cross-node reduction passed in job 3209888.
- Full pinned-data preparation completed in job 3209817, covering all 13 markets. No test-set evaluation was performed.
- Real-data smoke job 3210021 completed three updates and 48 examples with TP=8/DP=2, then saved and checked full sharded recovery state.
- Candidate 16,627,365,889 parameters: reserved 38,927,335,424 of 42,600,300,544 device bytes (about 91%); rejected on memory headroom and backup cost, not an OOM failure.
- Candidate 7,444,254,721 parameters: microbatch 16, reserved 32,904,314,880 device bytes, 2.189 seconds per synthetic microbatch on one TP node; selected for the deadline-bound pilot.
- Eight-shard upload measurement: 536,881,024 bytes in 12.919 seconds, about 41.56 MB/s; remote digests matched and temporary payloads were removed.
- Latest remains a full FP32 resumable state. Best inference matrices use BF16 with FP32 replicated normalization, reducing final transfer size to approximately 104 GB combined.
- Training cutoff moved to 15:00 UTC+02 to reserve roughly an hour for staging, upload, and verification. Larger intermediate models were not exhaustively searched; the backup deadline, not only GPU memory, limits this pilot.
- Dropout is zero and EMA is disabled for this capacity/operational pilot. Full-epoch coverage and full validation are not claimed; subset monitoring and wall-clock checkpoints provide pilot diagnostics only.

## Pilot Outcome and Operations

- Main job 3210061 received 12 nodes / 96 A100s, 384 CPUs and 3 TiB host RAM. Actual layout was TP=8, DP=12.
- It completed 21 updates / 6,720 real examples. The final update was at 14:59:32 UTC+02; final checkpoint/summary completion was about 15:03.
- Best and latest both reference step 21. Latest includes full FP32 optimizer and sampler state; best is an inference export.
- Fixed pilot subset: 839 anchors. Huber 5.023262 versus zero baseline 5.004593; MAE 5.490915 pp, RMSE 8.567953 pp. Prediction standard deviation was only 0.012205 pp versus target standard deviation 8.563569 pp. This does not establish useful forecasting skill.
- Full-run rank-0 reserved-memory peak was 37.775 GiB, above the one-node profiling estimate. Before longer runs, reduce microbatch and instrument all-rank allocated/reserved peaks.
- The main job ended FAILED/1 after artifacts were saved. Do not reclassify it as a clean job exit. The exact original shutdown cause could not be conclusively determined from the accessible terminal output.
- Added explicit process-group teardown, worker exception recording, and persistent per-rank logs. CPU TP=2/DP=2 and GPU TP=8/DP=12 DDP-gradient/lifecycle diagnostics passed; GPU retest job 3210183 exited 0. No further model training was performed after the cutoff.
- Initial final artifact publication verified 33 files totaling 104,331,123,677 bytes at 15:17:57 UTC+02, commit 6deab8dd68472a39d3ef24a34b30094dda61e8b7. The full transfer was substantially faster than the small probes; future planning should use representative full-checkpoint measurements plus contingency.
- Next research priority: a smaller matched-data baseline, longer and explicit example budgets, robust full validation, prediction-variance monitoring, and point-in-time data-quality checks. Do not scale model size again merely because memory is available.