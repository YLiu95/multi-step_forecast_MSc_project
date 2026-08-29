# New Experiment 1.2: Global Cross-Ticker Forecasting on TPU

## Open TensorBoard First

Current session link (temporary):

**https://magic-answer-coated-result.trycloudflare.com**

From the experiment directory, install the small tunnel binary once and start
TensorBoard plus a public Cloudflare tunnel:

```bash
cd /root/new_experiment_1.2_tpu
bash scripts/install_cloudflared.sh
bash scripts/start_tensorboard.sh
```

The second command prints a URL like:

```text
TensorBoard: https://example-words.trycloudflare.com
```

Check both the local service and its current public URL:

```bash
curl -I http://127.0.0.1:16006/
grep -o 'https://[-a-z0-9]*\.trycloudflare\.com' \
  /root/artifacts/new_experiment_1.2_tpu/tunnel/cloudflared.log | tail -1
```

The Cloudflare address is temporary. If the Kaggle session restarts, run the
start script again and use the newly printed address.

Prepare data and start training in separate terminals:

```bash
cd /root/new_experiment_1.2_tpu
python -m src.prepare_data
python -m src.benchmark --steps 5
python -m src.train --stop-after-epoch 5

# Resume after an interruption from the newest local or Hugging Face checkpoint
python -m src.train --resume --stop-after-epoch 10
```

## 1. Experiment Overview

This experiment asks:

> Can 256 recent daily returns from a changing basket of 64 stocks help predict
> the size and direction of one named stock's cumulative seven-session move?

It makes a controlled extension of experiment 1.1:

| Setting | Experiment 1.1 | Experiment 1.2 |
|---|---:|---:|
| Markets | United States | 13 global markets |
| Available series | 4,767 | 39,260 before window filtering |
| Fixed target set | 167 | Every split-eligible ticker |
| Framework | PyTorch on two T4 GPUs | JAX/Flax on TPU v5e-8 |
| Model parameters | 16.6M | about 294M |
| Input window | 256 sessions | 256 sessions |
| Patch length | 8 sessions | 8 sessions |
| Tickers per sample | 64 | 64 |
| Forecast horizon | 7 sessions | 7 sessions |
| Heads | magnitude + direction | magnitude + direction |

The scientific idea has not changed. The experiment changes data coverage and
capacity while retaining the target, loss, basket size, role embedding, and
hierarchical Transformer structure from 1.1.

## 2. What One Training Example Means

Imagine a small teaching basket rather than the real 64-ticker basket:

```text
input order:       [MSFT, JPM, AAPL, XOM]
target position:   2
target ticker:     AAPL
input shape:       4 tickers x 256 daily log returns
future AAPL moves: [+0.4%, -0.2%, +1.1%, +0.3%, -0.5%, +0.2%, +0.1%]
signed target:     +1.4%
magnitude target:  abs(+1.4%) = 1.4%
direction target:  1 (up)
```

In the real sample, AAPL appears exactly once among 63 context tickers. Its
position is randomized. A learned role embedding marks that position, while a
learned ticker embedding tells the model which company each series belongs to.

The model emits two values:

1. A non-negative estimate of the absolute seven-session move.
2. A direction logit, converted to an up probability with a sigmoid.

It does not predict seven separate prices.

## 3. Data Preparation in Detail

### 3.1 Universe

The source is the private Hugging Face dataset
`YL95/new_experiment_1-data`: 135,493,260 daily rows and 39,260 price series
from AU, CA, CH, CN, DE, FR, GB, HK, IN, JP, KR, NL, and US.

The completed preparation produced this measured coverage:

| Market | Series | Train targets | Validation targets | Test targets | Train anchors |
|---|---:|---:|---:|---:|---:|
| AU | 1,853 | 1,214 | 1,557 | 1,774 | 3,919,711 |
| CA | 2,514 | 1,553 | 2,049 | 2,407 | 4,532,941 |
| CH | 266 | 213 | 233 | 225 | 739,946 |
| CN | 6,861 | 3,354 | 4,990 | 6,463 | 8,173,565 |
| DE | 915 | 704 | 840 | 858 | 2,195,663 |
| FR | 630 | 492 | 563 | 605 | 1,697,980 |
| GB | 1,492 | 1,115 | 1,352 | 1,451 | 4,172,788 |
| HK | 2,541 | 1,608 | 2,143 | 2,413 | 4,070,600 |
| IN | 4,797 | 3,083 | 3,559 | 4,426 | 8,393,418 |
| JP | 3,762 | 3,008 | 3,384 | 3,712 | 10,342,687 |
| KR | 2,522 | 1,732 | 2,078 | 2,438 | 5,141,076 |
| NL | 166 | 75 | 104 | 125 | 43,398 |
| US | 10,941 | 4,817 | 6,767 | 9,591 | 16,435,193 |
| **Total** | **39,260** | **22,968** | **29,619** | **36,488** | **69,858,966** |

There are also 25,199,313 validation anchors and 29,137,723 test anchors,
for 124,196,002 legal examples in total. An *anchor* is one legal target ticker
and forecast date; randomized context baskets create many more possible inputs.
All seven Mag7 tickers were found. The prepared memory-mapped panel occupies
1.9 GB, and the train-only global return scale is 3.406079%.

Only these four columns are read:

```text
ticker | group | date | adj_close_clean
```

No open, high, low, volume, technical indicator, fundamental, or news feature
is given to the model.

### 3.2 Why Adjusted Close

Raw close can jump mechanically during a stock split. Adjusted close rewrites
historical prices so splits and dividends do not look like investment gains or
losses. This experiment uses `adj_close_clean`, whose documented bad-tick
corrections also remove some very large genuine moves. That is a limitation:
cleaner labels come at the cost of making extreme-market prediction easier.

### 3.3 Price to Daily Log Return

For adjusted prices $P_t$ and $P_{t-1}$:

$$
r_t = 100\log\left(\frac{P_t}{P_{t-1}}\right).
$$

For example, a move from 100 to 102 gives:

$$
100\log(102/100) \approx 1.98\%.
$$

Log returns are useful because consecutive returns add. Seven future returns
can therefore be summed directly. Prices themselves trend and change scale;
returns are much closer to a stable learning problem.

### 3.4 Missing Sessions and Market Calendars

Each sample is restricted to one market. Every market keeps its own trading
calendar: a US holiday is not treated as a missing Japanese observation.

This decision came from measurement. For six representative markets after
2015, the intersection of all trading calendars retained only 84.1% of the
union of dates. A global intersection would discard about one day in six. A
global union would require fake zero returns or substantially more masking.

A 256-session input is legal only when all 256 returns exist. A target is legal
only when the next seven returns also exist. A long gap can therefore never be
silently converted into zero or crossed by a window.

### 3.5 Chronological Splits

| Split | Legal anchor dates |
|---|---|
| Train | through 2018-12-31, with the target ending before the boundary |
| Validation | 2019-01-07 through 2022-12-24, approximately by market calendar |
| Test | from 2023-01-07 onward, approximately by market calendar |

The exact boundaries are measured in each market's trading sessions, not
calendar days. Seven sessions are embargoed around split boundaries. Inputs may
use older historical data because that information was already available at
forecast time; targets may never cross a boundary.

The test set is evaluated once after model selection. Looking at it while
tuning would turn it into another validation set.

### 3.6 Scaling

One standard deviation is computed from all finite **training-period** returns
across all markets. Each input is divided by that one number and clipped to
$[-8,8]$. Validation and test data never influence the scale.

One global scale deliberately retains cross-market and cross-stock volatility
differences. That matches experiment 1.1 and preserves volatility level, which
is one of the strongest learnable signals in daily returns. Labels remain in
raw percentage units, so an MAE of 300 basis points still means 3%.

### 3.7 Every Available Target

A ticker is eligible in any split where it has at least one legal 256+7 sample.
A 2021 IPO can therefore be a validation or test target even though it cannot
be a training target. TensorBoard separately reports the `always_eligible`
cohort, containing only tickers available in all three splits.

This distinction matters. A newly listed ticker's identity embedding may be
weakly trained or untrained, so overall validation measures a harder problem
than the always-eligible chart.

## 4. How the 64-Ticker Basket Is Sampled

For each training row:

1. Select a target uniformly from all target tickers eligible in that split.
2. Select one legal anchor session for that target.
3. Find tickers from the same market with complete 256-session histories there.
4. Draw 63 distinct context tickers.
5. Insert the target exactly once at a random position.
6. Shuffle the resulting order.

Changing the basket composition is real augmentation: the target interacts
with different peers on repeated visits.

Shuffling order alone is mathematically a no-op here. Cross-ticker attention
has no positional encoding, so permuting inputs permutes internal states in the
same way; selecting the marked target then gives the same answer. We still
shuffle as a defensive check against accidental ordering assumptions.

## 5. Model, Step by Step

The model is a hierarchical Patch Transformer.

1. Each 256-return history is divided into 32 non-overlapping 8-day patches.
2. A linear layer maps each patch to a 1,024-dimensional token.
3. Patch position, ticker identity, and target/context role embeddings are added.
4. Twelve temporal Transformer blocks process 32 tokens within each ticker.
5. Mean pooling creates one summary token per ticker.
6. Eight cross-ticker blocks let the 64 summaries exchange information.
7. The marked target state is concatenated with its ticker embedding.
8. Separate heads predict magnitude and direction.

Attention over all $64\times32=2048$ tokens at once would be unnecessarily
expensive. The hierarchy performs attention over length 32 and then length 64.

Ticker identity uses a learned embedding rather than a sinusoidal encoding.
Ticker IDs are categories: ticker 101 is not inherently between tickers 100 and
102. Sinusoids are appropriate for ordered positions, not arbitrary companies.

## 6. Two Losses and Live Weight Control

Let the future daily log returns be $r_{t+1},\ldots,r_{t+7}$:

$$
R_{t,7}=\sum_{k=1}^{7}r_{t+k},\qquad
y_{mag}=|R_{t,7}|,\qquad
y_{dir}=\mathbb{1}[R_{t,7}>0].
$$

Magnitude uses Huber loss with $\delta=1\%$. Direction uses binary
cross-entropy. Initially:

$$
L=0.7L_{Huber}+0.3L_{BCE}.
$$

To change weights at the next epoch boundary, edit `loss_weights.json`. The
trainer validates and normalizes the two values, then records the applied
weights in TensorBoard, history, and the experiment diary.

Do not react to one noisy epoch. Wait for at least three comparable validation
points and check all of these together:

- raw magnitude and direction losses;
- magnitude and direction head gradient norms;
- magnitude MAE versus the zero-magnitude baseline;
- direction accuracy versus majority-class accuracy;
- direction Brier score versus prevalence Brier score.

BCE near $\log(2)=0.693$ means the direction head is near an uninformative 50%
probability. Increasing its weight does not create signal; it only amplifies
its gradients.

## 7. Why This Size Fits the TPU

The selected model has exactly 296,377,346 parameters and uses all eight TPU
cores with global batch 320, or 40 samples per core. The final full-universe
benchmark, including EMA and both conditioned heads, measured 1.514 seconds per
update, 13,528 ticker histories per second, 9.33 GB HBM per core, and 34.1%
estimated MFU. CPU batch construction took only 0.081 seconds, so input loading
does not starve the TPU.

At 500 updates, compute takes about 12.6 minutes per epoch before validation
and backup time. A 60-epoch ceiling is therefore roughly 13-15 hours; early
stopping can end it sooner.

Activation checkpointing (`nn.remat`) recomputes intermediate activations
during the backward pass instead of storing all of them. Without it, compilation
requested up to 36 GB per 16.9 GB core and failed. With it, measured steady HBM
was about 7 GB/core before adding the EMA copy.

The goal is not 100% HBM usage. HBM is storage, not computation. Leaving
compile and allocator headroom prevents a nine-hour run from dying because one
later operation needs a temporary buffer.

## 8. Reading TensorBoard

Start with these charts:

| Chart | Meaning |
|---|---|
| `validation/loss` | Declared checkpoint objective; lower is better |
| `validation/magnitude_mae_bp` | Typical magnitude error; 100 bp = 1% |
| `validation/direction_accuracy` | Correct up/down fraction |
| `validation/majority_direction_accuracy` | Accuracy from always choosing the common class |
| `validation/direction_brier` | Probability calibration error; lower is better |
| `validation/predicted_magnitude_std_pct` | Near zero warns of constant predictions |
| `training/gradient_norm` | Persistent clipping can indicate an excessive learning rate |
| `gradnorm/magnitude_head` | Weighted magnitude-head gradient size |
| `gradnorm/direction_head` | Weighted direction-head gradient size |
| `performance/hbm_used_gb` | Memory occupied on the busiest TPU core |
| `performance/mfu_percent` | Estimated fraction of peak matrix compute used |

The same validation predictions create country charts under
`validation_country/<market>/...`, plus `validation_cohort/mag7/...` and
`validation_cohort/always_eligible/...`. Eight Mag7 rows are reserved in each
320-row validation batch, producing 640 examples over the default 80 batches.

## 9. Backups and Recovery

Every five epochs:

- the full model, EMA, optimizer, and step are uploaded to
  `YL95/new_experiment_1.2_tpu/checkpoints/epoch_NNN/`;
- only the newest two periodic checkpoints are retained locally and remotely;
- the best EMA parameters are updated under `best model/` whenever validation
  loss improves;
- code, resolved config, JSONL history, diary, and TensorBoard events are pushed
  to GitHub under `new experiment 1/new_experiment_1.2_tpu/`.

Resume with:

```bash
python -m src.train --resume
```

The trainer checks local storage first, then downloads the newest Hugging Face
checkpoint. Credentials come from `HF_TOKEN` and `GITHUB_TOKEN`, falling back to
Kaggle Secrets. Tokens are never written to experiment logs.

## 10. Commands and Checks

The corrected two-epoch pilot completed successfully. Over ten optimizer
updates, validation loss changed from 2.71770 to 2.51071 and magnitude MAE from
401.8 to 374.0 bp. Direction accuracy remained near chance, as expected at this
stage. A 4.742 GB full checkpoint restored successfully, verifying the recovery
path before the main run.

The first five full epochs also completed. Validation loss improved
monotonically from 2.56104 to 2.50838, magnitude MAE improved from 381.7 to
374.0 bp, and direction Brier improved from 0.24681 to 0.24223 versus its
0.24823 baseline. Because both heads improved without overfitting or memory
instability, the 0.7/0.3 weights and all optimizer settings are retained for
epochs 6–15.

```bash
# Run all CPU-safe and tiny-TPU tests
python -m pytest -q

# Build the full panels (resumable downloads)
python -m src.prepare_data

# Compile and benchmark the real production configuration
python -m src.benchmark --steps 5

# Two-epoch plumbing pilot
python -m src.train --smoke

# Main run or recovery
python -m src.train
python -m src.train --resume

# Recommended staged monitoring; this keeps the 60-epoch cosine schedule intact
python -m src.train --stop-after-epoch 5
python -m src.train --resume --stop-after-epoch 10

# Validation during development; test only once after model selection
python -m src.evaluate --split val
python -m src.evaluate --split test --batches 80
```

## 11. Important Limitations

1. The source contains survivors only. Delisted and failed companies are badly
   underrepresented, so financial conclusions will be optimistic.
2. `adj_close_clean` can replace genuine large moves as well as bad ticks.
3. Per-market baskets do not learn same-day cross-market relationships. That is
   deliberately postponed rather than hidden behind fake calendar values.
4. Direction prediction is close to the noise floor in experiment 1.1. A small
   apparent accuracy gain must be compared with class balance and Brier score.
5. This is an academic forecasting experiment, not trading advice or a
   survivorship-free backtest.

## 12. Project Layout

```text
new_experiment_1.2_tpu/
├── README.md
├── selfevo_process.md
├── loss_weights.json
├── requirements.txt
├── scripts/
│   ├── install_cloudflared.sh
│   └── start_tensorboard.sh
├── src/
│   ├── benchmark.py
│   ├── callbacks.py
│   ├── config.py
│   ├── dataset.py
│   ├── download.py
│   ├── engine.py
│   ├── evaluate.py
│   ├── hub.py
│   ├── losses.py
│   ├── model.py
│   ├── prepare_data.py
│   ├── secrets.py
│   └── train.py
└── tests/
```

Large arrays and checkpoints live under
`/root/artifacts/new_experiment_1.2_tpu/`, outside Git.