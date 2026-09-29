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