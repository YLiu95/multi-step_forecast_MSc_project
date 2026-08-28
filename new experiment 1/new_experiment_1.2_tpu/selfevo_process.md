# Self-Evolution Process

This is the experiment diary. Each iteration records evidence before action so
that decisions remain reproducible and are not explanations invented after the
result is known.

## Iteration 0: Hardware and Architecture Calibration

- Observation: The environment exposes eight TPU v5e cores with 16.91 GB HBM
  each, 96 CPU cores, and 377 GB CPU RAM.
- Observation: A 294M-parameter candidate with `d_model=1024`, 12 temporal
  blocks, 8 cross-ticker blocks, 64 tickers per sample, and global batch 320
  ran at 1.019 seconds per step, about 20,098 ticker histories per second and
  an estimated 50.6% model FLOP utilisation.
- Observation: The same family without activation checkpointing required up to
  36 GB per core and failed compilation. With `nn.remat`, steady measured HBM
  was about 7.0 GB per core. `nn.scan` reduced compilation from 262 to 15 seconds.
- Action: Use the 294M configuration, `nn.remat`, and `nn.scan`. Keep substantial
  compile-time headroom instead of targeting 100% HBM occupancy.
- Lesson: High memory use is not the same as high hardware use. MFU measures
  useful matrix computation; an out-of-memory run performs no useful work.

## Iteration 1: Implementation Contract

- Observation: Unit tests cover model outputs, non-negative magnitude,
  log-return calculation, missing-data gaps, temporal embargoes, compressed
  anchors, same-market baskets, exact target placement, deterministic batches,
  dual-task metrics, a compiled optimizer update, and checkpoint restoration.
- Observation: Uniform validation would select very few of the seven Mag7
  targets from a 39,260-series universe.
- Action: Reserve eight Mag7 target rows per 320-row stratified validation batch,
  while deriving all overall and subgroup charts from that same forward pass.
- Lesson: A requested subgroup chart needs enough observations to be readable;
  merely adding a metric name does not create useful evidence.