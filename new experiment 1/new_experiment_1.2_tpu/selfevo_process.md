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

## Iteration 5: First End-to-End Pilot and Control Isolation

- Observation: A ten-update, two-validation pilot completed end to end. Loss
  moved from 2.62342 to 2.42544 and magnitude MAE from 401.9 to 374.1 bp. This
  confirms training, EMA, validation slicing, TensorBoard, and 4.5 GB full-state
  checkpoint serialization work at production shape.
- Observation: The recorded weights were 0.667/0.333 rather than the intended
  0.7/0.3. A unit test had written its 2:1 fixture to the shared live-control
  file because that path was not configurable.
- Action: Make `loss_weights_path` configurable, give the test a temporary
  file, restore 0.7/0.3, preserve this pilot as `smoke_contaminated`, and rerun.
- Lesson: Tests must never mutate an operator control file. Recording resolved
  controls in every history row made the mistake visible immediately.

## Iteration 6: Corrected Two-Epoch Pilot

- Observation: With the intended 0.7 magnitude / 0.3 direction weights, ten
  production-shape updates reduced validation loss from 2.71770 to 2.51071 and
  magnitude MAE from 401.8 to 374.0 bp. Direction accuracy moved from 0.469 to
  0.461 against a validation up-rate near 0.464.
- Observation: The complete 4.742 GB model, EMA, optimizer, and step checkpoint
  restored correctly at epoch 2. The best EMA-only artifact is 1.186 GB.
- Action: Keep the initial loss weights and learning rate for the main run.
  Direction is still at the noise floor, but ten updates are far too few to
  distinguish weak signal from normal stochastic variation.
- Lesson: A smoke run proves plumbing and catches scale errors. It does not
  provide enough independent validation points for hyperparameter selection.
## Epoch 1

- Observation: train loss `3.22062`; validation loss `2.62342`; magnitude MAE `401.9` bp; direction accuracy `0.469`; epoch `1.6` minutes.
- Loss weights: magnitude `0.667`, direction `0.333`.
- Action: Keep the initial settings until at least three comparable validation points exist.
- Lesson: compare validation trends and naive baselines, not training loss alone.

## Epoch 2

- Observation: train loss `2.77456`; validation loss `2.42544`; magnitude MAE `374.1` bp; direction accuracy `0.452`; epoch `0.1` minutes.
- Loss weights: magnitude `0.667`, direction `0.333`.
- Action: Keep the current settings because the declared validation objective improved.
- Lesson: compare validation trends and naive baselines, not training loss alone.

## Epoch 1

- Observation: train loss `3.32449`; validation loss `2.71770`; magnitude MAE `401.8` bp; direction accuracy `0.469`; epoch `1.6` minutes.
- Loss weights: magnitude `0.700`, direction `0.300`.
- Action: Keep the initial settings until at least three comparable validation points exist.
- Lesson: compare validation trends and naive baselines, not training loss alone.

## Epoch 2

- Observation: train loss `2.91685`; validation loss `2.51071`; magnitude MAE `374.0` bp; direction accuracy `0.461`; epoch `0.1` minutes.
- Loss weights: magnitude `0.700`, direction `0.300`.
- Action: Keep the current settings because the declared validation objective improved.
- Lesson: compare validation trends and naive baselines, not training loss alone.

## Epoch 1

- Observation: train loss `2.59971`; validation loss `2.56104`; magnitude MAE `381.7` bp; direction accuracy `0.526`; epoch `14.6` minutes.
- Loss weights: magnitude `0.700`, direction `0.300`.
- Action: Keep the initial settings until at least three comparable validation points exist.
- Lesson: compare validation trends and naive baselines, not training loss alone.

## Epoch 2

- Observation: train loss `2.53951`; validation loss `2.53599`; magnitude MAE `377.7` bp; direction accuracy `0.533`; epoch `13.2` minutes.
- Loss weights: magnitude `0.700`, direction `0.300`.
- Action: Keep the current settings because the declared validation objective improved.
- Lesson: compare validation trends and naive baselines, not training loss alone.

## Epoch 3

- Observation: train loss `2.53970`; validation loss `2.52150`; magnitude MAE `375.8` bp; direction accuracy `0.543`; epoch `13.1` minutes.
- Loss weights: magnitude `0.700`, direction `0.300`.
- Action: Keep the current settings because the declared validation objective improved.
- Lesson: compare validation trends and naive baselines, not training loss alone.

## Epoch 4

- Observation: train loss `2.54396`; validation loss `2.51271`; magnitude MAE `374.7` bp; direction accuracy `0.545`; epoch `13.3` minutes.
- Loss weights: magnitude `0.700`, direction `0.300`.
- Action: Keep the current settings because the declared validation objective improved.
- Lesson: compare validation trends and naive baselines, not training loss alone.

## Epoch 5

- Observation: train loss `2.53653`; validation loss `2.50838`; magnitude MAE `374.0` bp; direction accuracy `0.545`; epoch `13.2` minutes.
- Loss weights: magnitude `0.700`, direction `0.300`.
- Action: Keep the current settings because the declared validation objective improved.
- Lesson: compare validation trends and naive baselines, not training loss alone.

## Iteration 7: Five-Epoch Hyperparameter Review

- Observation: Validation loss improved monotonically from 2.56104 at epoch 1
  to 2.50838 at epoch 5. Magnitude MAE fell from 381.7 to 374.0 bp and remained
  31.70% better than the zero-magnitude baseline.
- Observation: Direction BCE improved from 0.68615 to 0.67599. Brier score
  improved from 0.24681 to 0.24223 against a fixed prevalence baseline of
  0.24823. Predicted up probability converged from 0.5263 to 0.4548 while the
  actual up rate was 0.4579, showing better calibration rather than collapse.
- Observation: Pre-clipping gradient norm averaged 1.96 and exceeded the 1.0
  threshold in 100 of 125 logged steps. Despite that, losses were finite and
  smooth, HBM stayed below 9.39 GB/core, and validation improved every epoch.
- Decision: Retain 0.7/0.3 task weights. The direction head has smaller
  gradients, but its validation loss and Brier score are already improving;
  increasing its weight now would trade away magnitude learning without
  evidence of a stalled classification task.
- Decision: Retain peak learning rate 2e-4, gradient clip 1.0, dropout 0.15,
  weight decay 0.1, batch 320, and the existing model. Frequent clipping alone
  is not instability while train and validation trends remain smooth.
- Action: Resume the unchanged schedule from epoch 5 through epoch 60, as
  requested, with early stopping disabled for this continuation. Validation
  still selects the best model, so later regressions cannot replace it.
  Periodic recovery checkpoints continue every five epochs.
- Lesson: Gradient scale is diagnostic evidence, not an automatic instruction
  to rebalance tasks. Validation behavior decides whether intervention helps.

## Epoch 6

- Observation: train loss `2.54711`; validation loss `2.51705`; magnitude MAE `375.1` bp; direction accuracy `0.547`; epoch `14.7` minutes.
- Loss weights: magnitude `0.700`, direction `0.300`.
- Action: Continue unchanged; no single epoch justifies a hyperparameter intervention.
- Lesson: compare validation trends and naive baselines, not training loss alone.

## Epoch 7

- Observation: train loss `2.53346`; validation loss `2.52688`; magnitude MAE `376.6` bp; direction accuracy `0.542`; epoch `13.1` minutes.
- Loss weights: magnitude `0.700`, direction `0.300`.
- Action: Continue unchanged; no single epoch justifies a hyperparameter intervention.
- Lesson: compare validation trends and naive baselines, not training loss alone.

## Epoch 8

- Observation: train loss `2.50823`; validation loss `2.52127`; magnitude MAE `375.9` bp; direction accuracy `0.541`; epoch `13.1` minutes.
- Loss weights: magnitude `0.700`, direction `0.300`.
- Action: Continue unchanged; no single epoch justifies a hyperparameter intervention.
- Lesson: compare validation trends and naive baselines, not training loss alone.

## Epoch 9

- Observation: train loss `2.54752`; validation loss `2.51399`; magnitude MAE `374.9` bp; direction accuracy `0.542`; epoch `13.2` minutes.
- Loss weights: magnitude `0.700`, direction `0.300`.
- Action: Continue unchanged; no single epoch justifies a hyperparameter intervention.
- Lesson: compare validation trends and naive baselines, not training loss alone.

## Epoch 10

- Observation: train loss `2.52511`; validation loss `2.51366`; magnitude MAE `374.9` bp; direction accuracy `0.541`; epoch `13.1` minutes.
- Loss weights: magnitude `0.700`, direction `0.300`.
- Action: Continue unchanged; no single epoch justifies a hyperparameter intervention.
- Lesson: compare validation trends and naive baselines, not training loss alone.

## Epoch 11

- Observation: train loss `2.51453`; validation loss `2.51101`; magnitude MAE `374.7` bp; direction accuracy `0.541`; epoch `13.1` minutes.
- Loss weights: magnitude `0.700`, direction `0.300`.
- Action: Continue unchanged; no single epoch justifies a hyperparameter intervention.
- Lesson: compare validation trends and naive baselines, not training loss alone.

## Epoch 12

- Observation: train loss `2.50170`; validation loss `2.51082`; magnitude MAE `374.8` bp; direction accuracy `0.546`; epoch `13.1` minutes.
- Loss weights: magnitude `0.700`, direction `0.300`.
- Action: Continue unchanged; no single epoch justifies a hyperparameter intervention.
- Lesson: compare validation trends and naive baselines, not training loss alone.

## Epoch 13

- Observation: train loss `2.52232`; validation loss `2.50755`; magnitude MAE `374.4` bp; direction accuracy `0.546`; epoch `13.1` minutes.
- Loss weights: magnitude `0.700`, direction `0.300`.
- Action: Keep the current settings because the declared validation objective improved.
- Lesson: compare validation trends and naive baselines, not training loss alone.

## Epoch 14

- Observation: train loss `2.50080`; validation loss `2.49989`; magnitude MAE `373.3` bp; direction accuracy `0.547`; epoch `13.1` minutes.
- Loss weights: magnitude `0.700`, direction `0.300`.
- Action: Keep the current settings because the declared validation objective improved.
- Lesson: compare validation trends and naive baselines, not training loss alone.

## Epoch 15

- Observation: train loss `2.50965`; validation loss `2.49937`; magnitude MAE `373.2` bp; direction accuracy `0.546`; epoch `13.1` minutes.
- Loss weights: magnitude `0.700`, direction `0.300`.
- Action: Keep the current settings because the declared validation objective improved.
- Lesson: compare validation trends and naive baselines, not training loss alone.
