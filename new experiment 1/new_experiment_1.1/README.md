# New Experiment 1.1: Cross-Ticker Seven-Day Forecasting

## Open TensorBoard First

From the experiment directory:

```bash
cd "/root/multi-step_forecast_MSc_project/new experiment 1/new_experiment_1.1"
tensorboard --logdir /root/artifacts/new_experiment_1.1/runs \
  --host 0.0.0.0 --port 16006 --reload_multifile=true
```

Confirm that TensorBoard is healthy:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:16006/
curl -s http://127.0.0.1:16006/data/runs
```

The first command should print `200`; the second should list run names. In VS
Code, forward port `16006` from the **Ports** panel. If tunnel forwarding is not
available, use the Cloudflare method documented in the parent experiment.

Start or resume training in a second terminal:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=2 -m src.train

# Resume after an interruption
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=2 -m src.train --resume auto
```

## 1. What This Experiment Changes

Experiment 1.1 makes a small, controlled change from `new_experiment_1`:

| Setting | Experiment 1 | Experiment 1.1 |
|---|---:|---:|
| Input unit | one ticker with 18 features | 64 tickers with one feature each |
| Input feature | engineered OHLCV features | adjusted-close daily log return only |
| Window | 256 trading days | 256 trading days |
| Patch length | 8 days | 8 days |
| Forecast | 20 daily returns | one 7-day magnitude and one direction |
| Target tickers | all sampled names | fixed 167-name target set |
| Heads | one return-path head | magnitude head + direction head |
| GPUs | two T4s with DDP | two T4s with DDP |

The purpose is to ask a new question without changing everything at once:

> Can the recent returns of a broad basket help forecast the seven-day move of
> one named target stock?

## 2. The Data Universe and Its Limitation

The pipeline reads the official NASDAQ and NYSE symbol directories and found
5,235 currently listed common-stock candidates on this run. Yahoo Finance then
provided at least 300 adjusted-close observations for 4,767 of them.

This is **not** a true historical point-in-time universe. A company delisted in
2005 is absent from today's exchange directory. Calling this “all stocks that
existed since 1990” would hide survivorship bias. A rigorous version requires a
licensed delisting-inclusive source such as CRSP, Norgate, or Compustat. This
experiment uses all usable **currently listed** NASDAQ/NYSE stocks and records
that limitation in `meta.json`.

This is the only responsible free-data approximation. It should be stated in
the dissertation rather than silently ignored.

## 3. How the 167 Targets Are Chosen

The target set contains exactly:

| Group | Count | Selection |
|---|---:|---|
| Magnificent Seven | 7 | fixed |
| Other Nasdaq-100 | 10 | seeded random sample |
| Other S&P 500 | 50 | seeded random sample |
| Outside both indices | 100 | seeded random sample |

Groups are made mutually exclusive before sampling, so no ticker is counted
twice. Random selection uses seed `1337`. A candidate must have at least one
legal 256-day input plus 7-day target in **train, validation, and test**. This
prevents a newly listed company from occupying a target slot it cannot train.

The exact result is saved at:

```text
/root/artifacts/new_experiment_1.1/panel/target_tickers.csv
```

On this run, there are 736,609 train anchors, 166,165 validation anchors, and
150,103 test anchors across the 167 targets.

## 4. Data Preparation, Step by Step

Run once:

```bash
python -m src.prepare_data
```

### Step A: Download adjusted close only

For each ticker and date, only adjusted close is retained. Adjusted close
accounts for stock splits and dividends. A raw close can appear to fall by 50%
on a 2-for-1 split even though the investor did not lose 50%; adjusted close
removes that false event.

Downloads are split into 200-symbol parquet files. If the environment crashes
after chunk 18, the next run begins with chunk 19. This also avoids holding one
very large pandas table in CPU RAM.

### Step B: Convert price to daily log return

For adjusted prices $P_t$ and $P_{t-1}$:

$$
r_t = \log\left(\frac{P_t}{P_{t-1}}\right).
$$

The code stores percentage log return, $100r_t$, for readable scales. For
example, if adjusted close changes from 100 to 102:

$$
100\log(102/100) \approx 1.98\%.
$$

Multiplying by 100 changes units, not information. The model input is this one
series only. No open, high, low, volume, calendar, or technical indicator is
provided.

### Step C: Validate a 256-day window

An input window ending at day $t$ is valid only if all returns
$r_{t-255},\ldots,r_t$ exist. Missing values are never silently interpreted as
zero in a sampled window. A cumulative-sum test computes validity for the full
panel without creating every window.

Illustrative sample with four tickers:

```text
basket ticker IDs: [MSFT, XOM, AAPL, JPM]
target position:   2
input:             4 tickers x 256 daily log returns
target ticker:     AAPL
future returns:    [0.4%, -0.2%, 1.1%, 0.3%, -0.5%, 0.2%, 0.1%]
signed 7-day move: 1.4%
magnitude label:   abs(1.4%) = 1.4%
direction label:   1 because 1.4% > 0
```

The real basket contains 64 unique tickers. The target is inserted at a random
position, so the model cannot learn “column zero is always the answer.” Every
sample must contain its target exactly once.

### Step D: Split by time

| Split | Anchor period |
|---|---|
| Train | through 2018-12-31 |
| Validation | 2019 through 2022 |
| Test | 2023 onward |

The seven future returns of a train anchor must finish before the train cutoff.
A seven-trading-day embargo separates splits. A validation input may include
older training dates; that is legitimate past information, as in live use.

The test split must remain untouched while choosing hyperparameters. Repeatedly
checking test performance would turn the test set into another validation set.

## 5. Why 64 Tickers per Sample

There is no universal correct basket size. More tickers provide more market
context but increase activation memory and attention cost.

Measured on both 15.36 GB T4s:

| Per-GPU batch | Peak reserved VRAM | Throughput per GPU |
|---:|---:|---:|
| 64 | 10.67 GB | 62.6 samples/s |
| 80 | 13.17 GB | 77.3 samples/s |

The chosen setting is **64 tickers per sample and 80 samples per GPU**. Thus
each GPU processes 5,120 ticker histories per optimizer step. The remaining
memory covers the 0.22 GB resident panel, EMA weights, compiler workspace, and
allocator variation. Increasing again would trade crash resilience for little
scientific value.

## 6. Model in Plain Language

The model is a hierarchical Patch Transformer with 16.6 million parameters.

1. Each 256-day ticker history becomes 32 non-overlapping patches of 8 days.
2. Four temporal Transformer blocks learn patterns within each ticker.
3. Averaging its 32 patch tokens produces one summary token per ticker.
4. Four cross-ticker Transformer blocks let the 64 ticker summaries interact.
5. The token at the randomized target position is selected.
6. Two separate heads predict magnitude and direction.

This hierarchy avoids attention over all $64\times32=2048$ patch tokens at
once. Instead, temporal attention handles length 32 and cross-ticker attention
handles length 64. That is much cheaper and easier to fit on T4 GPUs.

## 7. Ticker Encoding Decision

Use a **learned embedding** for ticker identity, not a sinusoidal encoding.

A ticker ID is a category. If IDs 10, 11, and 12 represent unrelated firms,
there is no reason ticker 11 should be mathematically “between” 10 and 12.
Sinusoidal encodings impose an ordered geometry and are appropriate for time or
position, where nearby positions genuinely are related.

The model adds learned ticker identity to each ticker’s patch tokens. It also
concatenates the target ticker’s embedding directly with the selected target
state immediately before both heads. This explicit conditioning is a good
choice: it tells one shared model which company to predict. The **embedding is
not itself part of the numeric output**; it conditions the computation that
produces the output. Returning an embedding as an extra target would not help
unless there were a separate, justified representation-learning objective.

A learned role embedding also marks the target versus context tickers.

## 8. Two Targets and Two Losses

Let the seven future daily log returns be $r_{t+1},\ldots,r_{t+7}$. First form:

$$
R_{t,7}=\sum_{k=1}^{7}r_{t+k}.
$$

The two labels are:

$$
y_{mag}=|100R_{t,7}|, \qquad y_{dir}=\mathbb{1}[R_{t,7}>0].
$$

The magnitude head ends in `Softplus`, so it cannot predict a negative
magnitude. It uses Huber loss, which behaves like squared error for ordinary
mistakes but is less dominated by rare extreme moves. The direction head emits
a logit and uses binary cross-entropy.

$$
L = 0.7L_{Huber} + 0.3L_{BCE}.
$$

This initial weighting gives more emphasis to the richer continuous task while
retaining a meaningful direction gradient. Do not tune it from one epoch. If
one component remains flat over several epochs, compare gradient scale and
validation metrics before changing weights.

## 9. Reading TensorBoard

Start with these plots:

| Scalar | Interpretation |
|---|---|
| `validation/loss` | Main checkpoint and early-stopping criterion; lower is better |
| `validation/magnitude_mae_bp` | Typical magnitude error in basis points; 100 bp = 1% |
| `validation/direction_accuracy` | Fraction of correct up/down labels; compare with class balance |
| `validation/direction_brier` | Probability calibration error; lower is better |
| `validation/actual_up_rate` | Base rate needed to interpret accuracy |
| `validation/predicted_up_probability` | Detects collapse to always-up or always-down |
| `validation/predicted_magnitude_std_pct` | Detects a nearly constant magnitude output |
| `training/gradient_norm` | Repeated clipping at 1.0 suggests an excessive learning rate |
| `performance/fp16_loss_scale` | Repeated decreases indicate fp16 overflow |
| `performance/gpu_memory_reserved_GB` | Should remain flat; steady growth suggests a leak |

Direction accuracy of 50% is not automatically bad if the model has barely
trained, and 55% is not automatically good if 55% of labels are “up.” Compare
`direction_accuracy` with `actual_up_rate` and inspect Brier score.

The two-epoch pipeline pilot improved validation loss from 3.13036 to 3.07989
and magnitude MAE from 461.6 to 454.5 bp. Direction accuracy remained near
50%. Forty optimizer updates are enough to validate plumbing, not enough to
make a scientific conclusion or retune the model.

## 10. Files and Storage

```text
new_experiment_1.1/
├── README.md
├── selfevo_process.md
├── requirements.txt
└── src/
    ├── benchmark.py       two-GPU memory and throughput measurement
    ├── callbacks.py       TensorBoard, history, checkpoints, backups
    ├── config.py          every experiment setting
    ├── dataset.py         GPU-resident balanced basket sampler
    ├── download.py        universe, target strata, adjusted-close chunks
    ├── engine.py          DDP training, validation, EMA, resume
    ├── hub.py             GitHub and Hugging Face clients
    ├── losses.py          weighted dual loss and metrics
    ├── model.py           hierarchical Patch Transformer
    ├── prepare_data.py    end-to-end data preparation
    └── train.py           command-line training entry point
```

Large files stay outside Git:

```text
/root/artifacts/new_experiment_1.1/
├── cache/                       resumable parquet downloads
├── panel/                       compact arrays, metadata, exact target list
├── runs/<run>/tb/               TensorBoard events
├── runs/<run>/history.jsonl     one machine-readable row per epoch
└── checkpoints/<run>/           latest two local checkpoints and best model
```

## 11. Backups and Recovery

Every five epochs, rank 0:

1. uploads a full checkpoint to
   `YL95/new_experiment_1/new_experiment_1.1/checkpoints/`;
2. deletes older remote periodic checkpoints, retaining the latest two;
3. uploads the best weights immediately to
   `new_experiment_1.1/best model/best.pt`;
4. mirrors config/history into this GitHub folder and pushes it.

Credentials are read from `HF_TOKEN`/`GITHUB_TOKEN`, a protected local cache,
or Kaggle Secrets. Tokens are never written to logs. Backup exceptions are
reported but never terminate training.

After a crash, rerun data preparation (cached chunks are skipped), then:

```bash
torchrun --standalone --nproc_per_node=2 -m src.train --resume auto
```

## 12. Necessary Adjustments, and Nothing More

The experiment intentionally does not add news, fundamentals, intraday data,
sector labels, graph neural networks, or a third objective. Those may be useful
later, but each would make it harder to attribute an improvement.

The necessary adjustments were:

- balanced target sampling so long-history targets do not dominate;
- random target position plus role embedding;
- explicit learned ticker identity at the prediction heads;
- temporal then cross-ticker attention for tractable memory;
- chronological splits and a seven-day embargo;
- separate robust regression and classification losses;
- deterministic validation baskets, sharded across GPUs;
- measured batch sizing and frequent crash-safe backups.

The next decision should come from several full validation points recorded in
`selfevo_process.md`, not from adding more architecture in advance.