# Experiment Diary

## 2026-09-29: Authorized Pilot

- The user approved TP=8 per node and DP equal to node count, superseding the earlier DP=1 restriction.
- PyTorch 2.6.0 with CUDA 12.4 wheels was selected; shared tutorial software is unchanged.
- Source pinned to `a8e7a7dd55424b6c4d8de0553f9a1f43539a4b05`; data pinned to `bcbbefdbe2313673895eb1a0d354747a9f1624fa`.
- The user explicitly approved public publication of weights and aggregate metrics to the named Hugging Face repository. Raw data and credentials remain private.
- Latest training cutoff is 15:15 UTC+02, backup verification target 15:50, reservation ends 16:00. Measured transfer time may require an earlier stop.
- The pilot keeps the signed seven-session target and chronological split contract. A fixed pilot validation subset selects the pilot best checkpoint; this is not full-validation model selection, and no test metrics will be used.
- Model capacity must be measured including FP32 optimizer state, rematerialized activations, and backup cost. No largest-model or training-performance result is claimed yet.