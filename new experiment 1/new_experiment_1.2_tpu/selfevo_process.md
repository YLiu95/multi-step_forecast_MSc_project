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

## Iteration 2: Exact Training-Step Benchmark

- Observation: The exact Flax implementation, including AdamW state, EMA,
  dual conditioned heads, role embeddings, and activation rematerialization,
  compiled successfully on all eight TPU cores in 81.9 seconds.
- Observation: With the complete NL panel it held 256.3M parameters, used 8.29
  GB HBM/core, processed 13,976 ticker histories/second, and reached an
  estimated 35.2% MFU at global batch 320. The global ticker table adds about
  40M parameters, giving a projected 296.4M total.
- Action: Keep global batch 320 for the full-data benchmark. The observed
  memory leaves enough room for the global embedding and compiler variation.
- Lesson: Synthetic architectural probes are useful for choosing a region, but
  the exact optimizer, EMA, and heads must be benchmarked before a long run.

## Iteration 3: Complete Data Audit

- Observation: Preparation consumed all 135,493,260 source observations from
  all 39,260 series. It produced 69,858,966 train, 25,199,313 validation, and
  29,137,723 test anchors. The global train-only scale is 3.406079%.
- Observation: There are 22,968 train targets, 29,619 validation targets,
  36,488 test targets, and 22,859 targets eligible in every split. All seven
  Mag7 symbols are present. A production validation batch contained all 13
  countries and exactly eight reserved Mag7 rows.
- Action: Keep per-split eligibility and report the always-eligible cohort
  separately. Do not remove recent listings merely to make split membership
  identical.
- Lesson: "All tickers" cannot mean pretending every company existed in 1990.
  It means using each ticker wherever a legal historical window exists.

## Iteration 4: Full-Universe Hardware Check

- Observation: The exact 39,260-ticker model has 296,377,346 parameters. It
  compiled in 86.2 seconds and ran at 1.514 seconds/update, 13,528 ticker
  histories/second, 9.33 GB HBM/core, and 34.1% estimated MFU.
- Observation: CPU sampling took 0.081 seconds/batch, about 18 times faster than
  model updates, so the prefetch queue should keep the TPU supplied.
- Action: Retain global batch 320. The model uses all eight cores while leaving
  about 6 GB of physical HBM/core for compiler and runtime variation.
- Lesson: Measure the exact embedding table and optimizer state; extrapolation
  from a reduced-universe probe is useful but not sufficient.