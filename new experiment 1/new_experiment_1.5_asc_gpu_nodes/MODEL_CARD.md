---
library_name: pytorch
tags:
- time-series
- financial-forecasting
- tensor-parallel
- research
---

# Experiment 1.5: Athena GPU-Nodes Pilot

This is a short, experimental signed-return forecasting run, not a validated
trading model or a claim of predictive profitability. It predicts one signed
seven-market-session cumulative log return in percentage-point units.

## Architecture and Inputs

The model is a hierarchical temporal-then-cross-ticker Transformer with learned
ticker and target-role embeddings, 256-session histories, patches of eight, and
64 distinct within-market tickers per example. It has one unrestricted signed
regression head and uses Huber loss with delta 1.0.

Tensor parallelism is eight GPUs within a node; corresponding shards are data
parallel across nodes. The pilot uses BF16 matrix operations with FP32 parameters
and AdamW state, activation rematerialization, no EMA, and dropout zero. These
pilot choices are not a matched learning comparison against Experiment 1.2.

## Checkpoints

- `best/`: eight BF16 inference-matrix shards, with replicated normalization
  parameters retained in FP32. Selected on the fixed pilot validation subset.
- `latest/`: eight full FP32 model/optimizer shards and training/sampler metadata.
- `vocabulary.json`: the market/ticker identifiers needed to reconstruct inputs.
- `runs/`: aggregate TensorBoard events.
- `reports/`: data audit, capacity and transfer measurements, training summary,
  and remote backup verification.

Use the accompanying PyTorch model and TP loader in the public source directory:
https://github.com/YLiu95/multi-step_forecast_MSc_project/tree/main/new%20experiment%201/new_experiment_1.5_asc_gpu_nodes

Load `latest/tp-XX.pt` with `torch.load(..., map_location="cpu", weights_only=True)`;
do not load an arbitrary untrusted pickle. Reconstruct the recorded TP=8 layout
and use `ascgpu.train --resume` for continuation. Exact pilot continuation requires
the recorded DP layout and the private pinned dataset. Checkpoints do not merge
all GPU memory into a single device.

## Data and Limitations

Data comes from the private `YL95/new_experiment_1-data` revision
`bcbbefdbe2313673895eb1a0d354747a9f1624fa`, using `adj_close_clean`. No raw prices,
returns, sample-level records, or credentials are published here.

The training end date is 2018-12-31 and validation end date is 2022-12-31, using
the reference seven-session horizon and embargo masks. The test split is not
evaluated. Best selection is based on a small fixed market-stratified pilot
validation subset, not full validation. A short run and large parameter count do
not establish convergence, generalization, or profitability. Retrospectively
adjusted, cleaned data is not a point-in-time, survivorship-free backtest.

The current repository tree retains one best model and one latest checkpoint.
Git/Hugging Face history may retain older revisions.