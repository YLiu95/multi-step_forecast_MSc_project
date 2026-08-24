# New Experiment 1 — Multi-Step Equity Forecasting with a Patch Transformer

A rebuild of the multi-step forecasting pipeline for a scale it can actually
learn from: **2,524 US tickers, 1990–2026, 7.15 million training windows**, a
**38 M-parameter Transformer**, trained with **DDP across 2× Tesla T4** under
fp16 mixed precision.

```bash
cd "/root/new_experiment_1/new experiment 1"

python -m src.prepare_data                                        # build the dataset

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=2 -m src.train --run-name my_run        # train on both GPUs

tensorboard --logdir /root/artifacts/runs --port 6006 --bind_all  # watch it
```

**To monitor training, see [§7 Monitoring with TensorBoard](#7-monitoring-with-tensorboard)** —
in VS Code the quickest route is `Ctrl+Shift+P` → `Python: Launch TensorBoard` →
*Select another folder* → `/root/artifacts/runs`.

---

## 1. Overview

| | Previous notebook | This experiment |
|---|---|---|
| Universe | 2 tickers (AAPL, GOOG) | 2,524 liquid US tickers |
| History | 2015 → today | 1990 → today (9,227 trading days) |
| Training windows | ~2,100 | **7,153,696** |
| Input features | 2 (two close prices) | 18 (returns, volatility, momentum, volume, market context, calendar) |
| Target | raw price level, MinMax scaled | volatility-scaled **log returns**, 20 steps ahead |
| Model | LSTM(64) | PatchTST-style Transformer, 38.2 M params |
| Hardware use | ~15 % of one GPU | **100 % util, power-capped, on both GPUs** |
| Backups | none | GitHub (code + history) + Hugging Face (checkpoints) |

The headline problem with the previous setup was **ratio**: roughly 20,000
model parameters per training sample. Any model large enough to be interesting
could memorise the entire dataset. Everything below follows from fixing that.

---

## 2. The three corrections that matter

### 2.1 Forecast returns, not price levels

The previous pipeline fit a `MinMaxScaler` on the training slice and predicted
the *price*. That scaling is textbook-correct — and the learning problem is
still broken:

* Prices are **non-stationary**. They trend, so the mean and variance of the
  training period are not the mean and variance of the test period.
* The network's output range is calibrated to prices it saw in training. When
  the test period makes a new all-time high, the model is *structurally*
  incapable of predicting it.
* A \$3 stock and a \$500 stock cannot share a model at all.

We train on **volatility-scaled log returns** instead:

```
r_t     = log(C_t / C_{t-1})            stationary and additive
sigma_t = std(r_{t-59 .. t})            trailing, uses only data known at t
y_h     = r_{t+h} / sigma_t             the target, h = 1 .. 20
```

Because the *same* `sigma_t` divides every horizon step, the prediction is a
proper multi-step path that converts straight back to dollars:

```
P_{t+h} = C_t * exp( sigma_t * cumsum(y_hat)_h )
```

Dividing by `sigma_t` is what makes a mega-cap and a micro-cap contribute
comparable gradients — it is the reason one model can pool 2,524 tickers.

### 2.2 A Transformer, because an LSTM cannot fill a GPU

An LSTM over a 256-day window is 256 *sequentially dependent* small matrix
multiplies. Step `t+1` cannot start until step `t` finishes, so the GPU spends
most of its time on kernel-launch latency and its tensor cores idle. No batch
size fixes this — the dependency is in the architecture.

Patching removes it:

```
(B, 256, 18)  --reshape-->  (B, 32 patches, 8 days x 18 features)
              --linear--->  (B, 32, 512)
              --attention-> all 32 tokens processed in PARALLEL
```

Every layer becomes one large batched GEMM, which is exactly what fp16 tensor
cores exist for. Measured on one T4: **100 % utilisation at the 70 W power
cap**, i.e. the card is running as hard as it physically can.

Patching also cuts attention cost from `O(256²)` to `O(32²)` — a 64× saving
that is what makes a 12-layer model affordable.

### 2.3 Robust loss

Daily returns are heavy-tailed. MSE squares a 10-sigma earnings gap, so a
handful of days can dominate the gradient. We default to **Huber** (quadratic
near zero, linear in the tails) and ship **quantile / pinball** loss behind a
config flag, which predicts the 10th/50th/90th percentile and gives an honest
uncertainty band instead of a false point estimate.

---

## 3. Repository layout

```
new experiment 1/
├── README.md                  <- this file
├── requirements.txt
├── data prep and train 1.ipynb  <- data walkthrough + launches training
├── resume training.ipynb        <- resume from any checkpoint, compare runs
├── logs/<run_name>/             <- per-epoch history, committed automatically
├── tools/
│   └── make_resume_notebook.py  <- regenerates 'resume training.ipynb'
└── src/
    ├── config.py       every tunable knob, one dataclass, saved into each checkpoint
    ├── download.py     universe construction + chunked, resumable yfinance download
    ├── features.py     OHLCV -> 18 stationary features; anchor index + purged splits
    ├── prepare_data.py one-shot entrypoint: universe -> download -> features -> .npy
    ├── dataset.py      GPU-resident panel sampler (see §5)
    ├── model.py        PatchTST-style encoder
    ├── losses.py       Huber / MSE / pinball + finance metrics
    ├── engine.py       DDP, fp16 AMP, cosine schedule, EMA, checkpoint I/O
    ├── callbacks.py    TensorBoard, checkpointing, early stopping, backups
    ├── hub.py          Hugging Face + GitHub helpers (never raise)
    └── train.py        torchrun entrypoint
```

Large artefacts live **outside** the repo, in `/root/artifacts/`:

```
/root/artifacts/
├── cache/        parquet download chunks (resumable)
├── panel/        feat.npy, ret.npy, sig.npy, anchor_*.npy, meta.json
├── runs/<name>/  tb/  (TensorBoard)  +  history.jsonl  +  config.json
└── checkpoints/<name>/  ckpt_epoch_XXX.pt, latest.pt, best.pt
```

---

## 4. Data pipeline

**Universe.** Built from the official NASDAQ-traded symbol directory
(`nasdaqlisted.txt` + `otherlisted.txt`, 13,166 symbols), not from an index
membership list. This matters: today's S&P 500 contains only companies that
*survived*, so a model trained on it has never seen a firm collapse. Filtering
to common stock and clean tickers gives 5,306 candidates; a liquidity screen
(≥ 750 trading days of history, median dollar volume ≥ \$1 M) leaves **2,524**.

**Features** (18, all stationary and scale-free):

| Group | Features |
|---|---|
| Return / bar shape | `z_ret`, `z_gap`, `z_oc`, `z_range` |
| Volatility | `log_sigma` |
| Momentum | `z_mom5`, `z_mom20`, `z_mom60`, `z_mom120`, `z_dist252` |
| Volume | `vol_surprise` |
| Market context | `mkt_ret`, `mkt_mom20`, `breadth`, `dispersion` |
| Calendar | `dow`, `month_sin`, `month_cos` |

Momentum over `k` days is divided by `sigma * sqrt(k)`, the random-walk scaling,
so all look-backs are directly comparable. Market context is computed from the
panel's own cross-section rather than from SPY, so it exists back to 1990 with
no NaN gap. Every feature is then standardised with **train-period statistics
only** and winsorised at ±8 sigma.

**Splits** are chronological by anchor date:

| Split | Anchor dates | Windows |
|---|---|---|
| train | → 2018-11-29 | 7,153,696 |
| val | 2019-01-30 → 2022-12-01 | 1,956,782 |
| test | 2023-01-31 → | 2,172,838 |

The 20-bar purge gap is an **embargo**, not an overlap fix: a train anchor's
target window already ends inside the train period, so the target windows are
disjoint at zero purge. The embargo skips one full horizon because adjacent
target windows overlap each other and are therefore serially correlated.

We deliberately do *not* purge the full 256-day look-back. A validation input
window may contain training-period bars — that is just history, exactly what a
live model would have. Leakage means seeing a future **label**, not a past
**input**.

---

## 5. Why there is no `DataLoader`

The obvious design — a `Dataset` that slices a 256-day window, wrapped in a
multi-worker `DataLoader` — fails on this machine:

* Materialising every window is impossible: 7.15 M × 256 × 18 × 4 bytes ≈
  **130 GB**. There is 31 GB of RAM.
* Slicing lazily on CPU works, but feeding two T4s needs ~1 GB/s pushed through
  **4 CPU cores** and then over PCIe. The GPUs would starve, and worker
  processes are exactly what blows up RAM and kills a Kaggle session.

The whole feature panel is only `2,524 × 9,227 × 18 × 2 bytes (fp16) = 1.0 GB`.
So it is uploaded to VRAM **once**, and each batch is built by advanced-indexing
on the GPU:

```python
X = feat[ticker[:, None], t[:, None] + arange(-255, 1)]     # (B, 256, 18)
Y = ret [ticker[:, None], t[:, None] + arange(1, 21)] / sig[ticker, t]
```

Zero worker processes, zero host-to-device copies in the training loop, zero
CPU RAM pressure. This is the single most important engineering decision in the
project and it is what makes the 4-core constraint irrelevant.

---

## 6. Training setup

| | |
|---|---|
| Model | 38.2 M params — `d_model` 512, depth 12, 8 heads, `d_ff` 2048, pre-LN, GELU |
| Patching | 32 patches × 8 days, channel-mixing patch embedding |
| Head | flatten all 32 tokens → `Linear(16384, 20)` — direct multi-step, no autoregression |
| Precision | fp16 AMP + `GradScaler` (T4 is `sm_75`: fp16 tensor cores, **no bf16**) |
| Parallelism | `DistributedDataParallel`, one process per GPU, via `torchrun` |
| Batch | 1024 per GPU → 2048 effective |
| Optimiser | AdamW, β = (0.9, 0.95), wd 0.05 on matrices only |
| Schedule | 2-epoch linear warmup → cosine decay to 2 % of peak |
| Stabilisers | grad-clip 1.0, residual init scaled by `1/sqrt(2·depth)`, EMA 0.999 |
| Throughput | ~1,800 samples/s/GPU compiled (`torch.compile` = +23 %) |
| VRAM | ~9.7 GB model + 1.0 GB panel of 15.4 GB |

**Why `DistributedDataParallel` and not `nn.DataParallel`.** `DataParallel`
drives both GPUs from one Python process: the GIL serialises the forward
passes, gradients are gathered on GPU 0 so its memory fills first, and you
typically get ~1.3× instead of ~1.95×. DDP runs a real process per GPU and
overlaps the gradient all-reduce with the backward pass.

**Why warmup.** The attention softmax is very sensitive early in training; a
full-size first step can push it into a degenerate state it never leaves.

**Why `GradScaler`.** fp16 has a narrow exponent range. The scaler multiplies
the loss by a large factor before `backward()` so small gradients do not flush
to zero, then divides it out before the optimiser step.

---

## 7. Monitoring with TensorBoard

Logs are written to `/root/artifacts/runs/<run_name>/tb`. **Always point TensorBoard
at the parent directory**, `/root/artifacts/runs` — that makes it overlay *every*
run, past and present, on the same axes, which is the whole point of naming runs.

### Option A — VS Code's built-in panel (best experience)

1. Press **`Ctrl+Shift+P`** (`Cmd+Shift+P` on macOS) to open the Command Palette.
2. Type **`Python: Launch TensorBoard`** and press Enter.
3. Choose **"Select another folder"** and enter `/root/artifacts/runs`.

TensorBoard opens as a VS Code tab. Over a remote tunnel this needs no port
forwarding on your part — VS Code handles it.

### Option B — inline in a notebook

```python
%load_ext tensorboard
%tensorboard --logdir /root/artifacts/runs --port 6006
```

Re-run the cell to refresh. This is section 18 of `data prep and train 1.ipynb`
and section 4 of `resume training.ipynb`.

### Option C — a browser, via the tunnel

```bash
tensorboard --logdir /root/artifacts/runs --port 6006 --bind_all
```

Then open the **PORTS** panel in VS Code (next to TERMINAL). Port `6006` will be
listed — click the 🌐 globe icon in the *Forwarded Address* column. If it is not
listed, click **Forward a Port** and enter `6006`.

To leave it running after you disconnect:

```bash
nohup tensorboard --logdir /root/artifacts/runs --port 6006 --bind_all \
      > /root/artifacts/tensorboard.log 2>&1 &
```

### TensorBoard troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `TensorBoard could not bind to port 6006, it was already in use` | An instance is already running — very likely serving what you want | `curl -s localhost:6006/data/runs` to see what it has. Reuse it, or restart: `pkill -f "tensorboard.*6006"` then relaunch. Or just pick another port: `--port 6007` |
| Page loads but says **"No dashboards are active"** | Pointed at a run directory instead of the parent, **or** the log directory was deleted and recreated after TensorBoard started (it holds stale inodes) | Use `/root/artifacts/runs`, not `.../runs/<name>/tb`. If the path is right, restart TensorBoard |
| Scalars stop updating mid-run | Event file cached | Add `--reload_multifile=true`, or use the ⟳ refresh control (top right) |
| Port 6006 not in the VS Code **PORTS** panel | Not auto-detected | Click **Forward a Port** and enter `6006` |

Verify from the shell that it can actually see your run:

```bash
curl -s localhost:6006/data/runs          # -> ["patchtst_us_equities_v1/tb"]
```

An empty `[]` means TensorBoard found no event files — check the `--logdir` path.

### What to look at, in order of usefulness

| Tab / scalar | What it tells you |
|---|---|
| `val/std_ratio` | **Look here first.** Prediction std ÷ target std. If it slides toward 0 the model has given up and is predicting ≈ 0 for everything. A falling loss with a collapsing `std_ratio` is not learning — it is surrender. |
| `val/r2_vs_zero` | The honest score. **+0.005 … +0.02 is good.** Negative means worse than predicting "no change". |
| `val/rank_ic` | Spearman correlation of forecast vs outcome. 0.02–0.05 is a real signal. |
| `train/loss` vs `val/loss` | The classic generalisation gap. With 7 M samples and 38 M params it should stay narrow. |
| `train/grad_norm` | Should settle into a stable band. Repeated spikes at the clip threshold mean the LR is too high. |
| `train/amp_loss_scale` | fp16 health. It halves whenever a step overflows and is skipped. Occasional halving is normal; a scale that keeps collapsing means numerical trouble. |
| `perf/samples_per_sec` | Throughput (~3,050 total). A sudden drop means something else started competing for the GPUs. |
| `perf/gpu_mem_alloc_GB` | Should sit flat at ~10.7 GB. A slow climb means a memory leak. |
| **IMAGES** tab | `val/forecast_examples` — predicted vs actual cumulative 20-day paths, redrawn every epoch. The fastest way to see *how* the model is wrong, not just how much. |
| **HPARAMS** tab | Every run's config against its best metric, in one sortable table. |

> **Reading the loss curve honestly.** Validation loss here will improve by a few
> *thousandths* and then flatten. That is normal — daily equity returns are mostly
> unpredictable noise. If your loss drops dramatically, be suspicious before you
> are pleased, and go looking for look-ahead in your features.

**A resumed run continues the same curves.** `global_step` is restored from the
checkpoint, so new points land after the old ones rather than overwriting them
from step 0. Pass a different `--run-name` if you want a separate curve to
compare against instead.

**Not seeing anything?** The most common cause is pointing at a run directory
instead of the parent. Use `/root/artifacts/runs`, not
`/root/artifacts/runs/<name>/tb`.

---

## 8. Reading the metrics

`val/loss` is Huber on volatility-scaled returns. The interpretable numbers:

| Metric | What "good" looks like |
|---|---|
| `r2_vs_zero` | **+0.005 to +0.02 is a genuinely good result.** If you see 0.9, you have a look-ahead bug. |
| `rank_ic` | Spearman correlation, prediction vs outcome. 0.02–0.05 is a real signal in quant finance. |
| `dir_acc` | Direction of the cumulative 20-day move. 0.52–0.55 is meaningful. |
| `std_ratio` | Prediction std ÷ target std. **If this collapses toward 0 the model has given up and is predicting the mean.** Watch it. |

The baseline is *zero*, not the mean — "the price does not move" is the honest
naive forecast for a return series.

---

## 9. Reproducing

```bash
# 1. build the dataset (~9 min; download chunks are cached and resumable)
python -m src.prepare_data

# 2. train on both GPUs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=2 -m src.train --run-name patchtst_us_equities_v1

# 3. watch it (see section 7 for the VS Code and notebook alternatives)
tensorboard --logdir /root/artifacts/runs --port 6006 --bind_all

# 4. resume from the last checkpoint
torchrun --nproc_per_node=2 -m src.train \
    --run-name patchtst_us_equities_v1 --resume auto
```

Any config field can be overridden without editing a file:

```bash
torchrun --nproc_per_node=2 -m src.train --set depth=16 --set loss="'quantile'"
```

---

## 10. Backups

* **GitHub** — code, config and `logs/<run>/history.jsonl` are committed and
  pushed every 5 epochs and at the end of training.
* **Hugging Face** (`YL95/new_experiment_1`):
  * `checkpoints/` — full checkpoints (model + optimiser + scaler + EMA + RNG
    state) every 5 epochs, plus the run history for resuming.
  * `best model/best.pt` — weights only, uploaded the instant validation
    improves.

Every backup call swallows its own exceptions. A flaky network should cost you
one upload, not twelve hours of GPU time.

---

## 11. Suggested next steps

1. **Ablate the architecture.** Train the old LSTM on the *same* 7 M-window
   dataset. The comparison — same data, same loss, same budget — is the core
   result of an MSc chapter.
2. **Switch to quantile loss** (`--set loss="'quantile'"`) and report calibration:
   do 80 % of outcomes actually land inside the 10–90 band?
3. **Ablate the feature groups.** Drop market context, drop momentum, and
   measure `rank_ic`. This tells you where the signal really comes from.
4. **Regime analysis.** Break the test metrics down by year. Performance in
   2023–2026 will not be uniform, and explaining *when* the model works is more
   interesting than a single average.
5. **Address survivorship bias properly.** The screen still requires a ticker to
   be listed today. A point-in-time universe would be the rigorous fix.
