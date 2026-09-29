# Experiment 1.5: Athena GPU Plan

Adapted on 2026-09-29 from the supplied Experiment 1.5 draft, the Athena GPU inventory, and the public Experiment 1.2 source. This is an implementation and experiment plan, not a report of completed GPU benchmarks or training.

**Execution update, 2026-09-29:** the user subsequently approved PyTorch and TP=8 per node with DP across nodes. The resulting pilot is implemented in `projects/multi-step_forecast_MSc_project/new experiment 1/new_experiment_1.5_asc_gpu_nodes/`. It trained a 7.44B model on 12 nodes for 21 updates, and published best/latest checkpoints and events to `YL95/experiment-1.5-asc-gpu-nodes`. Its measured results, shutdown failure and diagnostic retest, and verified backups are recorded in that implementation's README and reports. Those records supersede the original framework, DP=1, and artifact-destination defaults below; the historical planning text is retained for provenance. This pilot did not complete a full epoch or full validation and did not beat its zero-return baseline.

## 1. Agreed Decisions

| Topic | Decision |
| --- | --- |
| Forecast | One signed seven-session cumulative log return for one target ticker. |
| Prediction head | One regression head; remove the separate direction head and the two-loss weighting mechanism. |
| Inputs | Adjusted-close-derived daily log returns; 256-session histories; non-overlapping patches of 8; learned ticker identities. |
| Training coverage | One epoch visits every eligible target-ticker/forecast-date anchor once. |
| Context sampling | Each ticker appears at most once in a sample. Context baskets may recur across anchors and epochs. |
| GPU comparison | Benchmark an approximately 296M-parameter baseline and a much larger model using pure tensor parallelism across all eight GPUs: TP=8, DP=1. |
| Validation | Fixed monitoring subset every 50,000 training examples; full validation after each complete training epoch. |
| Splits | Preserve and audit Experiment 1.2's chronological splits, calendars, and embargo. |
| Teaching | Explain consequential choices, ask understandable questions at unresolved decision gates, and record decisions before changing the experiment. |

Keep other scientific changes modest. In particular, retain the hierarchical temporal-then-cross-ticker Transformer and within-market baskets unless a measured problem requires an explicitly approved change. Moving to GPUs does not require rewriting the JAX/Flax model in PyTorch.

## 2. TensorBoard First

The implementation README must begin with working instructions for opening TensorBoard. Once the private environment and artifact directory described below exist, start TensorBoard on the JupyterHub/VS Code host, which can read the shared scratch logs without owning GPUs:

```bash
export ARTIFACT_ROOT="/net/tscratch/people/$(id -un)/experiments/experiment_1.5_gpu"
"$HOME/.venvs/experiment-1.5-gpu/bin/tensorboard" \
  --logdir "$ARTIFACT_ROOT/runs" --host 127.0.0.1 --port 16006
```

Forward port 16006 privately through VS Code, or use an authenticated JupyterHub proxy if that feature is available. Start TensorBoard on the host whose port is being forwarded, not on an unrelated compute node. Verify that the local service responds and that new training events appear through the chosen access route.

The original draft requested a Cloudflare link. Use a public Cloudflare tunnel only if cluster policy permits it and the user approves the visibility of the logs. Otherwise use private access. Do not invent a working URL, expose credentials, or claim that a tunnel has been tested when it has not. No live TensorBoard service is started by preparation of this document.

## 3. What the New Environment Actually Provides

| Item | Verified inventory fact and consequence |
| --- | --- |
| GPU allocation | Eight NVIDIA A100-SXM4-40GB devices on one node; 40,960 MiB, or 40 GiB, per device. |
| Aggregate memory | 320 GiB distributed across eight devices, not an automatically unified allocation. |
| Interconnect | Every GPU pair reported NV12 connectivity. This supports investigating fast tensor-parallel collectives; bandwidth and training performance were not measured. |
| GPU capability | Ampere, compute capability 8.0 / sm_80; BF16 is an appropriate precision candidate. |
| Software | Rocky Linux 9.7; driver 595.71.05; active CUDA toolkit 12.4.131; Slurm 23.11.7. |
| CUDA distinction | The driver's advertised CUDA 13.2 is a compatibility ceiling, not the active toolkit version. |
| Python | The inspected interpreter is Python 3.11.5 in a shared tutorial virtual environment. JAX, PyTorch, NumPy, and the other listed scientific packages were not installed there. |
| CPU and RAM | The whole node has 128 exposed logical CPUs and about 1,008 GiB RAM, but a job receives only the CPU and host RAM it requests and is granted. |
| Current notebook host | The VS Code/JupyterHub job on t0017 has zero allocated GPUs. A separate GPU allocation does not attach devices to that existing notebook kernel. |
| Storage | Home is NFS under /net/people; scratch is Lustre under /net/tscratch. Reported filesystem free space is not a personal quota or retention guarantee. |
| Access limits | The tutorial reservation was scheduled only for 2026-09-29, 09:00-16:00 scheduler-local time. Continued GPU access has not been established. |

The earlier TPU environment provided roughly 16 GiB per core, so 40 GiB is about 2.5 times that per-device capacity, not three times. More importantly, the earlier model was replicated across TPU cores. Pure tensor parallelism can distribute one much larger model across GPU memories; model size therefore cannot be inferred by simply multiplying 296M parameters by the HBM ratio. Activations, optimizer state, temporary buffers, and replicated tensors must also fit.

Do not reuse the inventory job's one CPU, 1 GiB RAM, or two-minute limit for training. Do not assume the sampled node t0020 will be the next allocated node. Four active 200 Gb/s InfiniBand ports were observed. Each large-model run in the agreed design uses one eight-GPU node; additional nodes can host independent trials as described in Section 4.4. A synchronized multi-node training run would require a separate parallelism decision and communication tests.

## 4. Project, Storage, and Software Setup

### 4.1 Locations and ownership

Create the GPU experiment under the existing GitHub repository at:

```text
new experiment 1/new_experiment_1.5_gpu/
```

Use the proposed Hugging Face model repository `YL95/new_experiment_1.5_gpu` for model artifacts. Neither remote destination is created by this document. Preserve the old TPU experiment and its checkpoints unchanged.

On Athena, keep the repository and small documentation in a user-owned home project directory. Use:

```text
Python environment: $HOME/.venvs/experiment-1.5-gpu
Large artifacts:    /net/tscratch/people/<remote-user>/experiments/experiment_1.5_gpu
Secrets, preferred: $HOME/.config/experiment-1.5/secrets.env
Secrets, accepted:  $HOME/.env, kept private and outside the cloned repository
```

The observed remote user was tutorial042; discover the actual current user rather than assuming it remains the same. Do not use the Kaggle-specific /root layout. Confirm scratch permissions, quota, retention, backup destinations, and enough space for several sharded checkpoints before training. Shared scratch is working storage, not the only backup.

Keep raw downloads, memory-mapped panels, caches, checkpoints, and TensorBoard events outside Git. Process data market by market and use bounded prefetching; do not materialize the full dataset or all possible baskets in Python lists or GPU memory. Put caches on an approved scratch path and keep authentication files separate from caches and artifact uploads.

### 4.2 A dedicated GPU environment

Create a user-owned virtual environment rather than modifying the shared tutorial environment. The observed Python executable can bootstrap it:

```bash
/net/software/trainings/nvidia-bootcamp/2025-06/multigpu/bin/python3 \
  -m venv "$HOME/.venvs/experiment-1.5-gpu"
```

Inspect Experiment 1.2's requirements, then resolve and pin a mutually compatible GPU stack: JAX, jaxlib and its GPU plugins, Flax, Optax, the checkpoint library, NumPy, pandas, PyArrow, Hugging Face tooling, TensorBoard, and python-dotenv if file-based secrets are used. Confirm support for Python 3.11 and A100 sm_80; do not copy TPU-only dependencies or blindly install the newest releases. Record the resolved versions and installation command in a GPU-specific lock/requirements file.

Prefer an explicitly selected, supported JAX CUDA wheel installation route. If using wheel-provided CUDA/cuDNN/NCCL, avoid accidentally overriding those libraries with inherited module library paths. If using local CUDA libraries instead, verify every required version, including cuDNN and NCCL; the loaded NCCL module alone is not proof of compatibility. Do not change shared cluster modules or the shared environment globally.

Verify the actual GPU job's interpreter, JAX backend, eight visible GPU devices, small BF16 matrix computation, and cross-device collective results. Record module names, package versions, driver, device topology, and relevant non-secret paths. Device enumeration alone is not a successful compute or communication test. Register a private Jupyter kernel only if desired; selecting it in the existing zero-GPU job still does not grant GPUs.

### 4.3 Launch inside Slurm

Use one node and one Python process that sees all eight allocated GPUs for the JAX device mesh. Do not launch eight independent copies of the single-process program. The starting request is 32 CPU cores and 128 GiB host RAM per node. These resources were granted in the readiness check in Section 4.4, but their adequacy for preparation or training has not been measured; they are not an entitlement to the full node.

The following is a Bash launch template for the future GPU-adapted benchmark. It is not a command to execute locally on Windows or against unmodified TPU-only code. Set an approved account, partition, time limit, code location, and implemented run configuration first:

```bash
: "${GPU_ACCOUNT:?Set an approved Slurm account}"
: "${GPU_PARTITION:?Set an approved GPU partition}"
: "${GPU_TIME:?Set an approved wall time}"
: "${EXPERIMENT_CODE:?Set the remote experiment directory}"
: "${RUN_CONFIG:?Set the benchmark configuration path}"
reservation_args=()
if [[ -n "${GPU_RESERVATION:-}" ]]; then
  reservation_args=(--reservation="$GPU_RESERVATION")
fi
cd "$EXPERIMENT_CODE"
env -u SLURM_MEM_PER_CPU \
    -u SLURM_MEM_PER_GPU \
    -u SLURM_MEM_PER_NODE \
    salloc --immediate=60 \
    --partition="$GPU_PARTITION" --account="$GPU_ACCOUNT" \
    "${reservation_args[@]}" \
    --nodes=1 --ntasks=1 --cpus-per-task=32 \
    --gpus=8 --mem=128G --time="$GPU_TIME" \
    --job-name=experiment-1.5-gpu \
    srun --ntasks=1 --cpus-per-task=32 \
    "$HOME/.venvs/experiment-1.5-gpu/bin/python" \
    -m src.benchmark --config "$RUN_CONFIG" --steps 5
```

The GPU adaptation must support the displayed benchmark interface before these commands are advertised as working. Five steps are a smoke check, not a sufficient performance comparison. Clear the inherited Slurm memory variables only for the new submission, as shown. Before reusing the tutorial reservation, inspect its current status and permitted resources; omit a reservation only when the selected account/partition allows ordinary access.

For long training, provide an equivalent sbatch script with logs, explicit resources, and application-level checkpoint/resume support. Request a pre-timeout signal with enough lead time for a measured checkpoint, and handle it by checkpointing at a safe training boundary. Do not rely on an open notebook, terminal, or automatic requeue to preserve state.

### 4.4 Multiple nodes and a one-minute allocation wait

**Recommended execution layout:** keep TP=8, DP=1 within each large-model trial and use extra nodes for independent trials. The baseline remains eight-way data parallel within its own node. This preserves the agreed scientific comparison; allocating more nodes does not itself change the model or synchronize its training.

After the correctness gates pass, place the baseline and large-candidate benchmarks on separate nodes. Use further nodes only for predefined configuration or seed trials within the approved benchmark budget. Each trial needs its own configuration, seed, sampler state, counters, checkpoints, and TensorBoard run directory. Replicating an identical run into the same artifact directory is not useful parallelism and can corrupt outputs.

Launch one Python worker per node. A future trial dispatcher can use `SLURM_PROCID` to choose a distinct trial, but it must explicitly keep each independent trial single-process from JAX's perspective. Do not automatically initialize one multi-host JAX cluster merely because Slurm started multiple workers. Each trial still visits its own eligible anchors according to Section 6.

If the goal is instead to accelerate **one shared training run** across N nodes, the natural alternative is TP=8 within each node and DP=N across nodes. That changes the agreed DP=1 constraint, global-batch accounting, gradient synchronization, and recovery requirements. Obtain approval and implement/test multi-host JAX initialization and collectives first. TP across all 8N GPUs is another scientific and performance change, not an automatic consequence of requesting N nodes.

#### Maximum-without-delay allocation

Slurm's node range requests as many nodes as possible within the range without delaying the job's start. The upper bound of 48 below is the current configured partition size, not a promise of 48 available nodes. `--immediate=60` limits allocation waiting to 60 seconds; `--time` separately limits runtime.

This runnable command performs GPU discovery only, with a five-minute runtime ceiling, and automatically releases the allocation when discovery finishes:

```bash
env -u SLURM_MEM_PER_CPU \
  -u SLURM_MEM_PER_GPU \
  -u SLURM_MEM_PER_NODE \
  -u SALLOC_USE_MIN_NODES \
  -u GITHUB_TOKEN -u GH_TOKEN -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN \
  salloc --immediate=60 \
  --partition=tutorial --account=tutorial \
  --reservation=training-multigpu \
  --nodes=1-48 --ntasks-per-node=1 --cpus-per-task=32 \
  --gpus-per-node=8 --mem=128G --time=00:05:00 \
  --job-name=experiment-1.5-readiness \
  srun --ntasks-per-node=1 --cpus-per-task=32 --label nvidia-smi -L
```

Do not set a fixed total task count of 48 when allowing the node count to vary. The task count follows the allocated node count through `--ntasks-per-node=1`; inspect `SLURM_JOB_NUM_NODES` and `SLURM_NTASKS` inside the allocation. Use `--gpus-per-node=8`, not `--gpus=8`, which would request eight GPUs for the entire job. The credential variables are removed only from this diagnostic subprocess because GPU discovery does not need them.

For real trials, complete CPU-side setup, data preparation, and the relevant correctness checks first. Then substitute the implemented per-node trial dispatcher for GPU discovery and choose an approved wall-time budget. A longer runtime or different resource request can yield fewer immediately available nodes. The dispatcher must run only the number of approved trials actually needed, even if additional hardware is free. No training dispatcher is implemented by this document.

#### Observed readiness result

On **2026-09-29 at approximately 13:48:56 UTC+02:00**, the maximum-without-delay request above, using a Python readiness probe in place of `nvidia-smi -L`, was granted as job `3209586`:

| Item | Observed result |
| --- | --- |
| Allocated nodes | 10: `t[0001-0005,0011-0014,0020]` |
| GPU total | 80 NVIDIA A100-SXM4-40GB devices |
| Per node | 8 GPUs, 32 allocated CPUs, 128 GiB host RAM |
| Worker layout | 10 tasks, one per node |
| Allocation wait setting | At most 60 seconds; the request succeeded |
| Runtime ceiling | 5 minutes; released immediately after the checks finished |
| Existing interpreter | Shared tutorial Python, not a private experiment environment |
| JAX readiness | `jax`, `jaxlib`, `flax`, and `optax` metadata absent in that interpreter |
| Planned private environment | Not present at the configured path |

Every worker verified eight visible A100s and a 32-CPU affinity. This was a successful resource-readiness check, **not** a JAX compute test, an inter-node collective test, a tensor-parallel benchmark, or model training. The allocation is no longer active. The current local workspace contains this plan, not a runnable GPU-adapted experiment. Pin and check out the source repository, build the private environment, and implement/test the adaptation before requesting training resources.

Ten nodes was the granted size for this request at this time, not a permanent limit or guaranteed size for later jobs. The training reservation was scheduled to end at 16:00 the same day; recheck access before every subsequent run.

## 5. Data and Temporal Split Contract

### 5.1 Data universe and historical evidence

Use all usable data from the private dataset `YL95/new_experiment_1-data`, not a fixed list of 167 targets. Inspect the actual dataset revision, schema, coverage, duplicate rows, missingness, price adjustments, and market calendars before preparation. Pin the dataset revision and record exclusions with reasons. No private dataset was downloaded for this planning task.

Experiment 1.2's README reports 39,260 series across 13 markets, 135,493,260 source rows, and these prepared anchor counts:

| Split | Historical eligible anchors |
| --- | ---: |
| Train | 69,858,966 |
| Validation | 25,199,313 |
| Test | 29,137,723 |

These are historical reported counts, not a fresh audit of the current dataset. Recompute them for the pinned revision and eligibility rules. Every split-eligible target must be represented; an unusable ticker must be reported, not silently replaced by fabricated observations.

Retain within-market baskets and each market's own calendar. At a forecast cutoff, every input ticker needs 256 consecutive valid market-session returns, and the target needs seven valid future market-session returns. A 256-return input needs 257 adjusted prices. Do not bridge missing sessions, forward-fill labels, insert artificial zero returns, or use future availability to choose context tickers. Changing to mixed-market baskets would require a separate decision about time zones and information availability.

The existing source uses `adj_close_clean`. Preserve that choice for comparability, while recording that cleaned data may remove genuine extreme moves and that a survivors-only universe can bias financial conclusions. Current retrospectively adjusted data is not automatically a point-in-time, survivorship-free trading backtest.

### 5.2 Exact split logic to preserve

Static inspection of Experiment 1.2 found:

```text
train_end = 2018-12-31
val_end = 2022-12-31
horizon = 7
embargo_sessions = 7
```

For a market's ordered session array, let `train_boundary` and `val_boundary` be the indices of the last dates on or before those respective end dates. The source's anchor masks are:

```text
train: anchor <= train_boundary - 7
val:   anchor >= train_boundary + 7
       and anchor <= val_boundary - 7
test:  anchor >= val_boundary + 7
```

Each anchor is additionally subject to the valid-history, valid-future, and sufficient-context filters. Preserve these session-index rules rather than substituting approximate calendar dates from the old README. Older observations may appear in validation/test input histories because they were known at the forecast cutoff; future labels may not cross their split boundary.

The existing preparation code computes one global standard deviation using finite returns dated on or before the training end date, divides inputs by that scale, and clips normalized inputs to [-8, 8]. Preserve this training-only fitting and record the resulting scale. Keep labels in raw log-return percentage-point units rather than clipping or fitting them on held-out data.

Add synthetic boundary and missing-session tests, then audit prepared dates and label endpoints in every market. Record exact first/last eligible anchors, target counts, exclusions, scaler provenance, and dataset revision. The source code was inspected for this plan; its tests and the private data pipeline have not been executed here. Evaluate the test split only after all model-selection decisions are frozen.

## 6. Samples, Epochs, and What Uniqueness Means

An anchor is one eligible target ticker at one forecast cutoff. A full training epoch visits every such training anchor once, in a reproducibly shuffled order. For each visit:

1. Use the anchor's target and market; gather only information available at that cutoff.
2. Draw 63 distinct context tickers with complete 256-session histories from that market, excluding the target.
3. Insert the target exactly once, giving 64 distinct input tickers.
4. Shuffle ticker order, preserve ticker IDs and the target marker, and attach the signed-return label.

Keep 64 tickers in both model-size candidates initially so the size comparison does not simultaneously change the input task. Changing this number requires a new eligibility audit because smaller markets or early dates may not have enough complete histories. Retain the target/context role marker and condition the prediction head on the target ticker embedding, not on an arbitrary context ticker.

A repeated ticker basket at another date contains different observations; a different target also defines a different prediction. Baskets may recur, including across epochs. Do not enumerate all baskets or maintain an ever-growing global set of previously used combinations. A fixed target with N eligible tickers and K inputs has C(N-1, K-1) possible baskets; this is finite but can be enormous. For illustration, C(100, 8) = 186,087,894,300 unordered baskets before considering dates or targets.

Use a seeded anchor iterator/permutation with bounded memory and resumable epoch/position state. Assign each anchor exactly once across data-parallel replicas in the baseline; the pure-TP model consumes one shared batch, not eight independently sampled batches. Handle the last partial batch without silently dropping anchors. Padding must have a loss/metric mask and must not count as observed samples.

A permutation of the same basket is not new independent data. Without cross-ticker positional encodings, the existing cross-ticker attention is permutation-equivariant and the correctly selected target prediction should be invariant. Retain shuffling as a check against accidental order dependence and add a permutation-invariance test.

Covering all anchors weights longer histories more heavily than the old uniform-target random sampler. Report per-ticker and per-country coverage and macro-averaged diagnostics alongside the anchor-weighted primary score; do not silently switch the training objective back to target-balanced sampling.

## 7. One Signed-Return Head

For adjusted close P and the target market's trading-session index:

$$
r_t = 100 \ln(P_t / P_{t-1}), \qquad
y_{t,7} = \sum_{j=1}^{7} r_{t+j} = 100 \ln(P_{t+7}/P_t).
$$

The factor 100 retains Experiment 1.2's log-return percentage-point convention. A label of +1 means a cumulative log return of +0.01, not a 100% simple return. Seven steps mean seven market trading sessions, not seven calendar days.

Use a single unrestricted linear output. Positive predictions indicate positive cumulative returns; negative predictions indicate negative returns. Do not apply abs, sigmoid, softplus, ReLU, or a non-negative constraint to this output. Call it the signed-return head, not the magnitude head.

Start with the existing robust Huber loss applied to the signed label, retaining delta = 1.0 log-return percentage point as a documented baseline. There is no direction-head BCE, no two-head loss-weight parameter, and no live loss-weight file. The old magnitude-plus-direction objective and new signed objective are not numerically comparable.

Retain learned ticker embeddings and target conditioning. Distinct ticker IDs receive distinct trainable rows with independent initialization; their learned vectors are not guaranteed to remain different or orthogonal. Similar vectors can reflect similar behavior. Monitor norms and similarities when diagnosing collapse, but do not add an embedding-separation penalty without evidence and approval. Explicitly report validation/test-only tickers whose embeddings received no training updates.

Report signed-return Huber loss, MAE, RMSE, prediction/target means and standard deviations, and a constant-zero forecast baseline. Include a training-only fitted constant baseline where useful. Convert percentage-point errors to basis points consistently: one percentage point is 100 basis points. Optional directional accuracy is derived from the sign of the regression output, not from a second head; fix a neutral-zero handling rule and report its denominator. Do not report Brier score or calibrated direction probabilities without a probabilistic model.

## 8. Baseline Versus a Much Larger Pure-TP Model

### 8.1 Model candidates

Both candidates use the new signed head, the same dataset revision, splits, anchor order, normalization, 64-ticker baskets, and 256/8 window/patch settings. Train from newly initialized parameters for the new experiment; old two-head TPU checkpoints are references, not interchangeable resume files.

| Configuration | GPU baseline | Initial large profiling candidate |
| --- | --- | --- |
| Hierarchy | Temporal blocks, pooling, cross-ticker blocks | Same hierarchy |
| Width | 1,024 | 3,072 |
| Attention heads | 16 | 48 |
| Feed-forward width | 4,096 | 12,288 |
| Temporal / cross-ticker blocks | 12 / 8 | 16 / 12 |
| Approximate parameters | About 296M before the head change | About 3.3B; measure exact count after instantiation |
| Parallelism | Eight-way data parallelism | Pure tensor parallelism: TP=8, DP=1 |

The larger configuration is a concrete starting point for profiling, not a measured fit or a guaranteed optimum. Its relevant widths and head count divide by eight. It is roughly an order of magnitude larger than the baseline, rather than mechanically scaling the baseline by 2.5. Count parameters and training state exactly, then reduce or increase capacity within a documented benchmark budget if measurements require it. Confirm the final production size with the user after presenting the results.

### 8.2 What pure tensor parallelism requires

Use an explicit eight-device tensor/model mesh in JAX. Shard attention projections and heads, feed-forward matrices, and suitable embedding/output dimensions so all eight GPUs cooperate on each model invocation. Preserve sharding through the optimizer update and align optimizer moments and EMA state with their parameter shards. Implement and verify the required reductions and gathers; simply making eight GPUs visible does not shard the model.

Pure TP means one data-parallel group, no FSDP and no pipeline-parallel stage in this large-model candidate. Small normalization parameters, biases, or some activations may legitimately be replicated; account for their memory. Do not silently substitute a 4-TP x 2-DP hybrid. If TP=8 is unsuitable, present measurements and ask before changing this user-selected constraint.

NVLink helps TP communication, but it does not eliminate collective latency or turn local allocations into a shared memory pool. Compare tiny-model outputs, loss, and gradients against an unsharded reference before scaling up. Check optimizer-state sharding and inspect per-device memory to detect accidental replication of large tensors.

Keep BF16 as a candidate compute dtype, with numerically appropriate FP32 loss reductions and optimizer state. Use rematerialization/activation checkpointing where needed. Measure allocated and peak reserved memory on every GPU, including initialization, forward/backward, optimizer update, EMA, validation, and checkpoint operations. Maintain approximately 15-20% practical memory headroom unless measurements justify a different margin; do not target 100% HBM occupancy.

### 8.3 Batch accounting and a fair benchmark

For the baseline, effective global batch = per-GPU microbatch x 8 data replicas x accumulation steps. For pure TP, effective global batch = shared microbatch x accumulation steps. Do not multiply the pure-TP sample count by eight: its GPUs work on the same examples.

Use 320 distinct examples per optimizer update as the initial matched comparison target, inherited from Experiment 1.2, and tune microbatch/accumulation separately for each layout. If this target is infeasible or inefficient, agree on and log a revised comparison. Keep the loss normalization consistent with the number of real examples, including accumulation and padding.

Separate compilation/initialization from timing. After warmup, time a sufficiently long steady-state interval, explicitly synchronizing JAX's asynchronous work. Measure end-to-end unique examples/second as well as device-only time, peak per-GPU memory, data-loading latency, communication cost, validation throughput, and checkpoint duration/size. Use repeated timings to distinguish a sustained result from startup noise. Report ticker histories/second only as an additional metric, never as though it were examples/second.

For learning comparisons, report both matched-example/update budgets and matched wall-clock budgets. A faster or larger model is not automatically a better predictor. Compare full-validation signed-return metrics to the baseline and zero-return forecast; keep the test set untouched. Do not claim TP scalability, model fit, speedup, or improved accuracy before measuring them.

## 9. Training Budget, Validation, and Monitoring

Experiment 1.2 used 500 sampled batches x 320 examples = 160,000 examples per nominal epoch. At the reported 69,858,966 training anchors, a true full epoch is about 437 times larger. Do not copy its 60-epoch ceiling, two-epoch warmup, early-stopping patience, or checkpoint timings unchanged.

After the throughput benchmark, present an estimated full-epoch time, full-validation time, checkpoint overhead, available Slurm wall-time, and proposed total training-example/update budget. Define warmup and learning-rate decay in optimizer updates or unique examples and persist the schedule on resume. Keep EMA, clipping, dropout, and optimizer defaults as starting references, but explain that changing update counts or batch size changes their effective behavior. Approval of a much larger model is not approval for an unbounded run.

### 9.1 Monitoring every 50,000 training examples

Create one fixed, reproducible monitoring subset that covers every validation-eligible ticker at least once and includes the available Mag 7 and country cohorts. Record its exact anchors, context seeds, size, and coverage. Additional stratification may help small cohorts, but its aggregate score must be clearly labeled as a subset diagnostic rather than the full-validation objective. Benchmark its cost too.

Trigger monitoring at the first completed optimizer update after each additional 50,000 real training examples. Log the actual sample count; a batch need not divide 50,000 exactly. Count examples globally once, not once per TP device, and exclude padding. Preserve the sample counter and next threshold through resume. If an epoch-end full pass coincides with a monitoring trigger, reuse the evaluation as appropriate rather than performing redundant work.

### 9.2 Full validation once per complete epoch

Visit every eligible validation anchor exactly once with deterministic context selection, consistent batching, and no random target subsampling or dropped tail. Full validation means full anchor coverage with a defined context per anchor, not enumeration of every possible basket. Aggregate sums and counts correctly so the final partial batch and unequal country sizes do not bias the score.

Use the full-validation signed-return Huber score as the declared best-checkpoint objective; use a consistent choice of EMA or non-EMA parameters for both candidates and record it. Preserve an end-of-run full pass if the budget stops between epochs, clearly labeling the fractional training coverage. The monitoring subset does not select the best model.

At historical sizes, full validation every 50,000 training examples would require about 504 validation examples per training example. That is why the user chose subset monitoring plus epoch-end full validation. Revised anchor counts and measured throughputs, not this historical ratio alone, determine the actual time budget.

### 9.3 TensorBoard groups and learning adjustments

| Tag group | Content |
| --- | --- |
| train/ | Signed Huber loss, learning rate, gradient norm, unique samples seen, optimizer updates, and fractional/full epoch progress. |
| monitor/ | Clearly labeled fixed-subset metrics and coverage. |
| validation/ | Full-pass signed Huber, MAE, RMSE, baseline errors, moments, counts, and optional derived sign accuracy. |
| validation_cohort/mag7/ | The available AAPL, AMZN, GOOGL, META, MSFT, NVDA, and TSLA targets, with counts and missing members reported. |
| validation_country/<market>/ | Separate country metrics and sample counts from the same evaluation predictions. |
| validation_cohort/always_eligible/ | Tickers eligible in train, validation, and test; also report untrained-identity cohorts separately. |
| performance/ | Distinct examples/second, optimizer-step time, peak memory per GPU, host RAM, data wait, and checkpoint/evaluation duration. |

Compute cohorts during the same pass instead of rereading the entire dataset per chart. Explain units, direction of improvement, denominators, baselines, and limitations for every published metric in the README. Weighted overall metrics and macro country/ticker metrics answer different questions and must be named accordingly.

Do not tune on the test split or react to one noisy monitoring point. Make one justified change at a time after inspecting comparable observations. Record the evidence, question, decision, configuration change, sample/update count, and subsequent outcome in the experiment diary named selfevo_process.md. Larger capacity cannot manufacture predictability in noisy financial returns.

## 10. Checkpoints, Backups, and Recovery

Keep the original five-full-epoch archival cadence and retain only the latest two archival checkpoints under the new model repository's checkpoints directory. Because a full epoch is now much longer, also maintain rolling recovery checkpoints on a measured wall-clock cadence, initially targeting every 30-60 minutes, and before a planned stop or Slurm timeout. A failed upload must not delete the last restorable backup.

Place recovery and archival checkpoints in distinct directories with explicit retention. Keep the latest two successful recovery checkpoints and the latest two archival checkpoints, plus a separate best model artifact. Confirm that quota and upload throughput can support this policy for the large model; adjust the time cadence with the user if transfers are too expensive.

Save all state required for an exact continuation: sharded parameters, optimizer moments, EMA, random generators, anchor permutation/iterator position, context-sampling state, epoch position, unique-example/update counters, next monitoring threshold, learning-rate schedule, best score, dataset revision, ticker vocabulary, scaler, resolved config, and software versions.

Use a compatible sharded JAX checkpoint mechanism. Avoid reconstructing all parameters and optimizer state on GPU 0 merely to save them. Write to a staging directory, complete all shards and metadata, verify the restore/manifest, then publish a completion marker and update the latest reference. Account for host-memory and disk needs as well as GPU memory. A checkpoint is not verified by the presence of a filename alone.

Keep the best full-validation model under the requested best model directory, including the inference parameters and everything needed to reconstruct inputs and outputs. Preserve raw versus EMA provenance and do not confuse an inference-only best artifact with a complete resumable training checkpoint.

The intended Hugging Face layout is:

```text
YL95/new_experiment_1.5_gpu/
  README.md
  config.json
  checkpoints/epoch_00005/
  recovery/latest-completed-checkpoints/
  best model/
  runs/<run-id>/events.out.tfevents.*
```

Flush and synchronize TensorBoard event files under runs so they remain available after the cluster session ends. Verify their visibility on the Hugging Face repository; do not assume uploading a file automatically verifies the hosted viewer. Back up code, configuration, small histories, teaching documentation, and the diary to GitHub often; put large checkpoints and event files on Hugging Face. Do not upload the private raw dataset to the public code repository. Confirm the visibility of new artifact repositories before publishing potentially sensitive material.

Prove recovery before the main run: stop a short trial, restore it, and compare the next anchors, labels, predictions, optimizer counters, and schedule to an uninterrupted control. Re-test with the actual TP=8 sharding and a representative checkpoint size before trusting multi-hour training.

## 11. Secrets in JupyterHub

Yes: if cluster policy permits local secret files, a private .env-style file is a reasonable fallback when the site does not provide an approved secret-injection mechanism. It is plaintext, not encrypted storage. Put it outside the repository, notebook folders, shared project directories, caches, and backup trees. Do not place it inside the virtual environment itself.

The requested location `/net/people/tutorial/tutorial042/.env` is acceptable under those conditions. It already existed; only its ownership and permissions were inspected. Its mode was tightened from `644` to **`600`**, and the home directory was mode `700`. No token values were opened, printed, or validated. Keep this file outside the future cloned repository and exclude it from Git, backups, and artifact uploads; permissions do not prevent tools running as your own user from reading it.

Verify metadata without displaying contents:

```bash
chmod 600 "$HOME/.env"
stat -c '%a %U %G %n' "$HOME/.env"
```

Use the remote user's private configuration directory. Create it with restrictive permissions, create or transfer the file privately, and then verify permissions without printing its contents:

```bash
umask 077
mkdir -p "$HOME/.config/experiment-1.5"
chmod 700 "$HOME/.config/experiment-1.5"
chmod 600 "$HOME/.config/experiment-1.5/secrets.env"
```

The final chmod command is run after the private file exists. Populate it through a trusted local editor or secure transfer, not by pasting tokens into chat, shell command arguments, a notebook cell, or a committed document. The local Windows secret file does not automatically appear in the remote JupyterHub environment.

Use the variable names GITHUB_TOKEN and HF_TOKEN. Prefer fine-grained, minimally scoped tokens: access to the private source dataset and write access only to the intended model repository, and GitHub contents access only to the required repository. Set appropriate expiry and rotate any exposed token.

An .env file is not automatically loaded by JupyterHub, Python, or Slurm. Load it explicitly in the component that needs remote access, without overwriting already injected environment variables. For the requested home-directory file, after installing python-dotenv in the private environment:

```python
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(
  Path.home() / ".env",
    override=False,
)
```

If using the preferred private configuration directory instead, change the path to `Path.home() / ".config" / "experiment-1.5" / "secrets.env"`. Choose one explicit location; do not search parent directories for arbitrary dotenv files or execute the file as shell code.

Check only whether required variables are present; never print values. Read/download/upload components need credentials, not every pure computation component. Ensure a separately submitted Slurm job can load the intended private file or receives the approved injected variables; setting values in one notebook's Python process does not configure a separate batch job.

Never write credentials into Git remotes, URL query strings, Slurm scripts, command-line flags, resolved configuration, checkpoint metadata, TensorBoard, exception dumps, or the diary. Exclude .env-style files and secrets.env from repository and backup rules as defense in depth. Do not dump the full process environment for reproducibility. Unix permissions do not protect tokens from cluster administrators; use only a cluster and storage policy you trust.

No token values are needed to revise this plan. The local secret file was not opened, and no credentials are included in either document.

## 12. Teaching and Execution Gates

Teach at an undergraduate deep-learning level using concrete samples and short explanations. Distinguish known facts, hypotheses, proposed defaults, and measured results. When asking a question, explain what changes depending on the answer, the cost or risk, and a recommended option. Do not silently reinterpret an ambiguous requirement.

Before moving between stages, present the evidence and resolve only the questions that materially affect the next stage:

1. Access and security: confirm a valid allocation, wall-time budget, quota/retention, private artifact visibility, and the permitted TensorBoard access route.
2. Data audit: preserve the split rules, pin the revision, verify eligibility/counts, and approve any unavoidable exclusions or data-quality changes.
3. Correctness: pass signed-target, split-boundary, missing-data, complete-epoch, sample-count, permutation, and tiny TP equivalence tests.
4. Hardware benchmark: establish eight-GPU compute/collective health, profile the baseline and large pure-TP candidate, and measure validation/checkpoint overhead.
5. Production choice: explain the measured memory/throughput tradeoff, select the final model and sample/update budget with the user, and verify recovery.
6. Training and reporting: monitor without test leakage, record interventions, run full validation, then evaluate the frozen selected model on the test split and verify remote artifacts.

Produce executable source and ordinary Markdown documentation for implementation; no new notebook is required. Include clear sample construction, signed-label calculations, architecture diagrams where useful, metric definitions, benchmark results, launch/resume instructions, and known data limitations. Never claim that larger hardware, a larger parameter count, or a successful import proves better generalization.

## 13. Evidence and Remaining Verification

The hardware facts come from the user-supplied Athena inventory captured on 2026-09-29 at 12:33:31 UTC+02:00 in diagnostic job 3209119. That allocation was released and is not a current training session.

The following public Experiment 1.2 sources were inspected on 2026-09-29:

- [Experiment directory and README](https://github.com/YLiu95/multi-step_forecast_MSc_project/tree/main/new%20experiment%201/new_experiment_1.2_tpu): historical counts, architecture, TPU measurements, and limitations.
- [Configuration](https://github.com/YLiu95/multi-step_forecast_MSc_project/blob/main/new%20experiment%201/new_experiment_1.2_tpu/src/config.py): dates, horizon, embargo, batch size, epoch length, and model settings.
- [Data preparation](https://github.com/YLiu95/multi-step_forecast_MSc_project/blob/main/new%20experiment%201/new_experiment_1.2_tpu/src/prepare_data.py): split masks, validity filters, context eligibility, and training-period scaling.
- [Sampler](https://github.com/YLiu95/multi-step_forecast_MSc_project/blob/main/new%20experiment%201/new_experiment_1.2_tpu/src/dataset.py): distinct context selection, target inclusion, signed labels, and fixed-count random batch generation.

These links follow the mutable main branch; pin an exact source commit before implementation. No software installation, source tests, private dataset access, model benchmarks, remote repository creation, or artifact uploads have been performed for this plan. The later allocation-readiness check and token-file permission change are recorded in Sections 4.4 and 11; they do not satisfy the remaining experiment execution gates.