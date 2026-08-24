# Self-Evolution Process

This file is the experiment diary. It records evidence before action so that
hyperparameter changes remain explainable and reproducible.

## Iteration 0: Architecture and Capacity Check

- Observation: A synthetic two-GPU run of the full architecture at batch 64 per
  GPU used 10.05 GB allocated / 10.67 GB reserved on each 15.36 GB T4 and
  processed 62.6 samples/s/GPU.
- Action: Tested batch 80 per GPU because the first run had safe headroom.
- Observation: Batch 80 used 12.48-12.55 GB allocated / 13.17 GB reserved and
  processed 77.3 samples/s/GPU. Both DDP ranks reported the same throughput.
- Decision: Set batch 80 per GPU (effective batch 160). Do not increase further:
  the remaining approximately 2.2 GB must cover the return panel, EMA copy,
  compiler workspace, and normal allocator variation during a long run.
- Lesson: GPU capacity should be chosen by peak-memory measurement, not by model
  parameter count alone. Activations dominate this architecture because each
  sample contains 64 ticker histories.

## Iteration 1: Real-Data Pipeline Pilot

The first pilot completed epoch 1 successfully but exposed a callback message
list that was not recreated after the console consumed it. This was an
experiment-control bug, not a model failure. Message creation now uses
`setdefault`, and the unchanged two-epoch pilot completed cleanly.

### Pilot Epoch 1

- Observation: train loss 2.73099; validation loss 3.13036; direction accuracy 0.500; magnitude MAE 461.6 bp; peak allocated VRAM 12.79 GB on rank 0.
- Action: Keep the initial settings until a trend is visible; one epoch is not evidence for retuning.
- Lesson: Decisions require validation trends, not training loss alone.

### Pilot Epoch 2

- Observation: train loss 2.48354; validation loss 3.07989; direction accuracy 0.499; magnitude MAE 454.5 bp; peak allocated VRAM 12.79 GB on rank 0.
- Action: No automatic hyperparameter change; continue collecting comparable validation points.
- Lesson: Decisions require validation trends, not training loss alone.

The pilot is too short to justify changing the learning rate or 0.7/0.3 task
weights. The full run starts from the original defaults so this experiment has
one controlled change set rather than tuning against 800 validation examples.

## Iteration 2: Five-Epoch Calibration

- Observation: Across 500 updates, validation loss improved monotonically from
  3.16945 to 2.82465 and magnitude MAE improved from 467.6 to 419.1 bp.
- Observation: Predicted magnitude standard deviation rose from 0.23% to 1.02%,
  so the regression head is learning variation rather than collapsing to a
  constant output.
- Observation: Direction BCE remained near the no-skill value
  $\log(2)=0.693$ and accuracy varied from 0.491 to 0.507 against an actual up
  rate of 0.520.
- Decision: Keep the model and 0.7/0.3 weights unchanged. Increasing direction
  weight before its validation BCE improves would magnify noise, not signal.
- Decision: Start the long run from epoch 0 under a new run name. The
  calibration used a five-epoch cosine schedule, so resuming its optimizer at
  epoch 5 under a 60-epoch schedule would create an artificial learning-rate
  discontinuity.
- Lesson: Output spread and a task-specific naive baseline are needed to tell
  genuine learning from a falling aggregate loss.

## Epoch 1

- Observation: train loss 3.04801; validation loss 2.85530; direction accuracy 0.575; magnitude MAE 421.6 bp; peak allocated VRAM 12.28 GB on rank 0.
- Action: Keep the initial settings until a trend is visible; one epoch is not evidence for retuning.
- Lesson: Decisions require validation trends, not training loss alone.

## Epoch 1

- Observation: train loss 2.66483; validation loss 3.16945; direction accuracy 0.500; magnitude MAE 467.6 bp; peak allocated VRAM 12.28 GB on rank 0.
- Action: Keep the initial settings until a trend is visible; one epoch is not evidence for retuning.
- Lesson: Decisions require validation trends, not training loss alone.

## Epoch 2

- Observation: train loss 2.38180; validation loss 3.00990; direction accuracy 0.507; magnitude MAE 445.3 bp; peak allocated VRAM 12.28 GB on rank 0.
- Action: Generalization gap is widening; prefer stronger regularization or earlier stopping.
- Lesson: Decisions require validation trends, not training loss alone.

## Epoch 3

- Observation: train loss 2.41002; validation loss 2.93292; direction accuracy 0.499; magnitude MAE 434.4 bp; peak allocated VRAM 12.28 GB on rank 0.
- Action: No automatic hyperparameter change; continue collecting comparable validation points.
- Lesson: Decisions require validation trends, not training loss alone.

## Epoch 4

- Observation: train loss 2.39974; validation loss 2.88126; direction accuracy 0.491; magnitude MAE 427.1 bp; peak allocated VRAM 12.28 GB on rank 0.
- Action: No automatic hyperparameter change; continue collecting comparable validation points.
- Lesson: Decisions require validation trends, not training loss alone.

## Epoch 5

- Observation: train loss 2.40153; validation loss 2.82465; direction accuracy 0.498; magnitude MAE 419.1 bp; peak allocated VRAM 12.28 GB on rank 0.
- Action: No automatic hyperparameter change; continue collecting comparable validation points.
- Lesson: Decisions require validation trends, not training loss alone.

## Epoch 1

- Observation: train loss 2.52148; validation loss 2.98761; direction accuracy 0.505; magnitude MAE 442.3 bp; peak allocated VRAM 12.28 GB on rank 0.
- Action: Keep the initial settings until a trend is visible; one epoch is not evidence for retuning.
- Lesson: Decisions require validation trends, not training loss alone.

## Epoch 2

- Observation: train loss 2.38089; validation loss 2.72217; direction accuracy 0.517; magnitude MAE 404.6 bp; peak allocated VRAM 12.28 GB on rank 0.
- Action: No automatic hyperparameter change; continue collecting comparable validation points.
- Lesson: Decisions require validation trends, not training loss alone.

## Epoch 3

- Observation: train loss 2.34806; validation loss 2.59145; direction accuracy 0.522; magnitude MAE 385.9 bp; peak allocated VRAM 12.28 GB on rank 0.
- Action: No automatic hyperparameter change; continue collecting comparable validation points.
- Lesson: Decisions require validation trends, not training loss alone.

## Epoch 4

- Observation: train loss 2.38750; validation loss 2.51974; direction accuracy 0.526; magnitude MAE 375.8 bp; peak allocated VRAM 12.28 GB on rank 0.
- Action: No automatic hyperparameter change; continue collecting comparable validation points.
- Lesson: Decisions require validation trends, not training loss alone.

## Epoch 5

- Observation: train loss 2.33307; validation loss 2.46900; direction accuracy 0.523; magnitude MAE 368.7 bp; peak allocated VRAM 12.28 GB on rank 0.
- Action: No automatic hyperparameter change; continue collecting comparable validation points.
- Lesson: Decisions require validation trends, not training loss alone.
