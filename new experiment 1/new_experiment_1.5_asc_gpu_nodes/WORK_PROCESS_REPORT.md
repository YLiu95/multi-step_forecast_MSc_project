# Experiment 1.5: End-to-End Working Process Report

**Date:** 2026-09-29. **Timezone:** UTC+02:00 unless explicitly stated otherwise.

This report documents the work from discovering GPU access through implementing,
testing, running, and backing up the Athena pilot. It also records the subsequent
shutdown diagnostic and reservation transition. This document was prepared after
the pilot closed; its publication is separate from the backups verified before
the original 16:00 deadline.

## 1. Executive Summary

A **7,444,254,721-parameter PyTorch model** was trained with **TP=8, DP=12 on
96 NVIDIA A100 GPUs**. The bounded pilot completed **21 optimizer updates and
6,720 unique training examples**, not a full epoch. Its best and latest checkpoint
selections both came from update 21.

The main job returned **FAILED / exit 1 after complete checkpoints, TensorBoard
events, and the training summary had been written**. The exact original exit-error
cause was not conclusively established. Process-group teardown was hardened, and
a subsequent tiny TP=8/DP=12 diagnostic, including DDP-gradient checks, exited
cleanly. The full training run was not repeated after that change.

Approximately **104 GB** of model artifacts were published to Hugging Face.
The final artifact set contained **37 verified payload files**, plus a separately
verified verification report. Code, configuration, dependency versions, and reports
were pushed to GitHub. Final cross-remote verification completed at **15:53:17**,
before the original reservation ended at **16:00**.

The pilot **did not demonstrate forecasting improvement**: signed Huber loss on
the fixed validation subset was **5.023262**, compared with **5.004593** for a
constant-zero forecast. The operational workflow was exercised, but convergence,
full-validation quality, and a globally maximum feasible model size were not established.

## 2. Evidence and Provenance

This report summarizes recorded commands, commits, tests, and saved artifacts.
The underlying evidence is retained in:

- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md): original planning document and its execution-update notice.
- [selfevo_process.md](selfevo_process.md): chronological decisions and observations.
- [reports/data_audit.json](reports/data_audit.json): dataset revision, markets, eligibility counts, and scaler provenance.
- [reports/training_summary.json](reports/training_summary.json): final counters, configuration provenance, and validation metrics.
- [reports/history.jsonl](reports/history.jsonl): aggregate optimizer-step history.
- [reports/operational_status.json](reports/operational_status.json): original job failure and the scope of the later diagnostic.
- [reports/backup_verification.json](reports/backup_verification.json): pinned artifact commit, sizes, and digests.
- [reports/runtime_manifest.json](reports/runtime_manifest.json) and [requirements.lock.txt](requirements.lock.txt): installed software versions.

| Reference | Recorded value |
| --- | --- |
| Original Experiment 1.2 source commit | `a8e7a7dd55424b6c4d8de0553f9a1f43539a4b05` |
| Private dataset | `YL95/new_experiment_1-data` |
| Dataset revision | `bcbbefdbe2313673895eb1a0d354747a9f1624fa` |
| Code commit used by the main training run | `bdadbe9175a135304679872e78ce2f34363dcaab` |
| Code/report snapshot verified at pilot closure | `755625ac3ca5078bc8739dc2e051de12d914bde1` |
| Final payload verification commit on Hugging Face | `0f40d433d2d7e017edf1489f8a2792a4462d29d6` |
| Hugging Face head including the verification report at closure | `fdf32358664e1244ac90194fbf12115a2cfffc1b` |

These are historical snapshots. Publishing this report creates a later GitHub
commit; it does not change the training-source commit or the saved model state.

## 3. Discovering and Understanding GPU Access

The initial VS Code tunnel ran inside JupyterHub job `3208358` on `t0017`.
That job had **zero allocated GPUs**, even though the cluster contained GPU nodes.
Consequently, local `nvidia-smi` reported no devices. The user's successful batch
benchmark was evidence that GPU access was mediated by Slurm, not that GPUs were
attached to the existing notebook kernel.

The requested notebook was found outside the home workspace, under the shared
scratch lab tree. Its saved output and examples were useful starting clues, but
fresh scheduled diagnostics were used to establish actual access.

Separate jobs verified two GPUs, then eight GPUs on one node. The hardware
inventory identified a **Supermicro AS -4124GO-NART+**, not an NVIDIA-branded DGX.
Its eight A100-SXM4-40GB GPUs and all-pairs `NV12` topology matched the notebook's
description of a DGX A100-like Athena node.

Key environment facts were:

- Eight GPUs per node, each reporting 40,960 MiB memory and compute capability 8.0.
- NVLink connectivity within the sampled node; this was initially topology evidence, not a bandwidth result.
- Two AMD EPYC 7742 sockets, 128 exposed logical CPUs, and approximately 1,008 GiB Linux-visible host RAM.
- Rocky Linux 9.7, Slurm 23.11.7, NVIDIA driver 595.71.05.
- The driver advertised CUDA 13.2 compatibility, while the active compiler toolkit was CUDA 12.4.131.
- Four active 200 Gb/s InfiniBand ports on the sampled node. Gb/s must not be confused with GB/s.

Host capacity was kept separate from job entitlement. The inventory's one CPU,
1 GiB RAM, and short time limit were diagnostic requests, not suitable training
resources. GPU memory was not treated as a single automatically pooled allocation.

### Scheduling and concurrency lessons

The partition advertised 48 configured nodes with no partition-level `MaxNodes`
cap, but availability was constrained by failures, other jobs, reservations, and
resource shape. A shared 15-node training reservation was not a personal allocation.
Its `FLEX` flag also meant that its node count was not a hard job-size ceiling.

Historical `sbatch --test-only` probes accepted a 29-node request with an estimated
start on October 1, while 30 nodes was rejected. That was a scheduling forecast
under one lightweight request, not an account quota or 29 nodes available immediately.
The earlier 12-node forecast was also a predicted start time, not a held allocation.

For real maximum-without-delay requests, the workflow switched to
`--nodes=1-48 --ntasks-per-node=1 --gpus-per-node=8` with `salloc --immediate=60`.
Slurm could select the largest count it could start without delaying initiation.
A fixed total of 48 tasks was deliberately not imposed on a variable node range.
Allocation waiting and job runtime were treated as separate limits.

Independent jobs can coexist, but their resources count toward applicable aggregate
limits. Busy resources can leave additional jobs pending, and submission limits can
reject requests. The visible user/QoS records showed no explicit concurrency cap;
unreturned parent-account limits were not assumed absent. Splitting work into jobs
was not presented as a way to bypass limits.

## 4. How the Experiment Scope Evolved

| Stage | Decision |
| --- | --- |
| Initial exploration | Establish GPU and multi-node access without modifying notebooks or starting training. |
| Original plan | Compare a roughly 296M baseline with a larger model, initially using TP=8 and DP=1 for the large candidate. |
| Multi-node clarification | Independent per-node trials were initially recommended to preserve that DP=1 constraint. |
| Explicit user authorization | Use TP=8 within each node and DP equal to node count; choose a suitable framework and make engineering decisions. |
| Implementation choice | PyTorch with CUDA 12.4 wheels, tensor-sharded model layers, and DDP across corresponding shards on different nodes. |
| Publication decision | The user explicitly approved publishing weights and aggregate metrics to the existing public Hugging Face target despite the dataset being private. |
| Time-budget decision | Run a bounded pilot and reserve time for verified best/latest backups rather than train until the reservation expired. |

The roughly 296M baseline and initial roughly 3.3B planning candidate were **not
trained in this session**. The numerical baseline reported for the actual pilot
was the constant-zero return forecast. The earlier planning defaults should not
be mistaken for completed experiments.

## 5. Repository, Environment, and Security Work

The existing repository was checked out sparsely and pinned before implementation.
The new work was placed in its own experiment directory; the prior TPU experiment
and its checkpoints were left unchanged. A private virtual environment was created
instead of modifying the shared tutorial interpreter.

The installed stack included PyTorch **2.6.0+cu124**, NumPy, pandas, PyArrow,
Hugging Face tooling, TensorBoard, safetensors, and pytest. Direct dependencies are
in [requirements.txt](requirements.txt); the full installed snapshot is in
[requirements.lock.txt](requirements.lock.txt).

Several concrete failures shaped the setup:

| Observation or failure | Response and lesson |
| --- | --- |
| Local GPU discovery returned no devices | Run GPU work in its own Slurm allocation; a new allocation does not grant GPUs to the existing notebook kernel. |
| `srun` rejected mutually exclusive memory settings | Clear inherited `SLURM_MEM_PER_CPU`, `SLURM_MEM_PER_GPU`, and `SLURM_MEM_PER_NODE` only for the new submission. |
| Clearing all library paths broke shared Python loading | Preserve the shared Python runtime path. |
| Keeping only Python's path then broke OpenSSL/PyArrow imports | Filter competing GPU-library paths, not all non-system library paths. Preserve Python and OpenSSL dependencies. |
| The VS Code task runner did not start preparation | Submit a separate CPU-only Slurm preparation job. |
| A QoS output-field alias was unsupported | Read this Slurm installation's default QoS fields instead of inferring limits from a failed query. |
| Port 16006 was occupied | Start the new TensorBoard server on 16007 without disturbing the existing service. |
| Terminal activity could change the effective working directory | Explicitly enter the experiment directory and source its environment in operational commands. |

[env.sh](env.sh) implements the library-path policy. It filters inherited
CUDA/NCCL/NVHPC-related paths while retaining required non-GPU libraries, allowing
the PyTorch wheel stack to supply its compatible CUDA libraries.

The existing home dotenv file was changed from mode 644 to **600**; its parent
home directory was mode 700. Tokens were loaded only by authorized local processes.
Their values were not displayed, committed, or placed in remote URLs, job arguments,
TensorBoard, checkpoints, or reports. Git authentication used transient process
configuration, and staged content was checked for credential values and prohibited
large/private artifacts before publication.

Raw data, prepared panels, caches, checkpoints, and events lived on user-owned
scratch storage, with the experiment artifact root restricted to mode 700. The
reported scratch hard quota was approximately 12 TiB; free filesystem space was
not represented as a personal entitlement. Raw data and sample-level records were
excluded from public uploads.

## 6. Data Preparation and Scientific Contract

Preparation used the pinned private dataset across all **13 markets and 39,260
series**. CPU-only job `3209817` completed in **72 seconds**, without occupying GPUs.
The resulting training index contained **69,858,966 eligible anchors**.

The implementation preserved these defining choices:

1. Use each market's ordered session calendar and `adj_close_clean` prices.
2. Convert prices to daily log-return percentage points; reject invalid prices and duplicate ticker/session observations rather than silently repairing them.
3. Require 256 valid daily returns, which need 257 prices, and seven valid future target returns.
4. Build each sample from 64 distinct within-market tickers, including the target exactly once.
5. Choose context eligibility using histories available at the cutoff, not future context availability.
6. Fit normalization using training-period returns only; clip normalized inputs, not the signed target.
7. Preserve the reference session-index splits and seven-session embargo.

The label is:

$$
y_{t,7}=100\log(P_{t+7}/P_t)=\sum_{j=1}^{7}100\log(P_{t+j}/P_{t+j-1}).
$$

It is a signed cumulative log return, not an absolute magnitude or a calibrated
direction probability. The model has one unrestricted regression output and Huber
loss with delta 1.0 in these percentage-point units.

The source boundaries were retained: training end 2018-12-31, validation end
2022-12-31, horizon seven sessions, and embargo seven sessions. Training labels
could not cross their boundary, and held-out data was not used to fit the scaler.
The test split was indexed during preparation but not evaluated or used for tuning.

Data was processed market by market and stored in memory-mapped arrays. Raw returns
were stored as float32 and normalized inputs as float16. A reproducible uint32
anchor permutation provided an epoch order without materializing all ticker baskets.
Context selection was deterministic from the recorded seed and anchor identity.
Padding was masked and excluded from losses and sample counts.

The pilot stopped partway through that order. Preparing all eligible anchors is
not the same as training on all of them. The fixed pilot validation subset contained
**839 anchors**, stratified by market with available Mag 7 targets; it did not cover
every validation ticker or constitute full validation.

## 7. Model and Parallel Execution

[ascgpu/model.py](ascgpu/model.py) implements the temporal-then-cross-ticker
Transformer. Temporal attention operates on non-overlapping patches, followed by
temporal pooling and cross-ticker attention. Ticker identity and target-role
information condition a single signed-return head.

The tensor-parallel implementation shards attention projections, attention heads,
feed-forward matrices, and embedding dimensions. Row-parallel reductions and
column-parallel input-gradient reductions are explicit. Small replicated parameters
are accounted for when computing the global gradient norm.

[node_entry.sh](node_entry.sh) launches one `torchrun` agent per Slurm node and eight
GPU workers per agent. In the main run, 12 nodes produced 96 ranks. Corresponding
TP shards participate in data-parallel gradient averaging across nodes. DP replicas
do not multiply the logical model's parameter count: this remained one **7.44B**
model, not a 12-times-larger model.

The selected configuration was width 4096, 64 attention heads, feed-forward width
16384, 20 temporal blocks, and 16 cross-ticker blocks. Training used BF16 matrix
computation with FP32 parameters and AdamW moments, activation rematerialization,
gradient clipping at 1.0, and an initial learning-rate warmup to `2e-5`.

Dropout was zero and EMA was disabled for this short capacity/operational pilot.
These departures were recorded; no matched learning comparison against the original
TPU experiment was claimed.

### Global-batch accounting

The effective global batch was **320 real examples**, not 320 per GPU. With 12 DP
replicas and microbatch 16, two microsteps provided 384 execution slots. The extra
64 slots were padding with zero loss weight. TP ranks within a node processed the
same examples cooperatively; only DP replicas contributed different examples.

Consequently, 21 updates represented exactly **6,720 examples**, not that count
multiplied by eight or by 96. The sampler position, epoch position, counters, and
next monitoring threshold were checkpointed for continuation.

## 8. Validation Gates and Capacity Selection

### Correctness checks

Eight unit tests covered signed labels, missing sessions, split boundaries,
timezone-explicit deadlines, deterministic/distinct-ticker sampling, padding,
artifact allowlisting, and CPU model/optimizer checkpoint restoration.

Distributed checks compared the TP implementation against an unsharded version
of the **new PyTorch signed-head model**, not against the old two-head Flax model.
They also checked ticker-permutation invariance and cross-node reductions.

| Check | Recorded result |
| --- | --- |
| CPU TP reference comparison | Maximum forward error about `2.38e-7`; gradient error about `5.96e-7` |
| GPU TP=8, DP=2 BF16 diagnostic | Passed; forward error 0.0078125 and gradient error 0.0126953125 within the diagnostic's BF16 tolerance |
| Real-data TP=8, DP=2 smoke test | Three updates, 48 examples, validation, and complete sharded recovery-state serialization |
| Post-change TP=8, DP=12 diagnostic | DDP-gradient error about 0.0021159; clean shutdown and exit 0 |

CPU and BF16 GPU tolerances differed deliberately. These were not claims of
bitwise equality. The CPU restore test reloaded model outputs and optimizer state;
large saved shards were checked for readable format, expected keys, optimizer-state
presence, and counters. A full 7.44B GPU resume-and-next-update equivalence test was
not performed.

### Capacity and backup tradeoff

| Candidate | Microbatch | Peak reserved bytes | Decision |
| --- | ---: | ---: | --- |
| 16,627,365,889 parameters | 1 | 38,927,335,424 | About 91% of reported device capacity; rejected on headroom and projected backup cost |
| 7,444,254,721 parameters | 16 | 32,904,314,880 | Passed the one-node headroom target and selected for the bounded pilot |

The probes included forward/backward computation, FP32 optimizer-state allocation,
and gradient clipping. The recorded 7.44B synthetic microbatch time was about
2.189 seconds. These short probes were capacity checks, not a statistically robust
performance comparison, and used synthetic inputs rather than claiming financial training.

The full TP+DP run later recorded **37.775 GiB peak reserved memory on rank 0**,
which was tighter than the single-node estimate. This is reserved memory, not an
all-rank measurement of live allocations. The lesson is to profile the actual DP
layout and reduce microbatch size before a longer run, not to assume the initial
headroom estimate remained valid.

Transfer probes also influenced the decision. A roughly 268 MB single-file probe
measured about 21 MB/s; an approximately 537 MB eight-shard probe measured about
42 MB/s and passed remote digest checks. Fresh synthetic payloads were used and
removed afterward. Those small probes were conservative planning signals, not a
guarantee of sustained large-checkpoint bandwidth.

The final large artifact publication was substantially faster: approximately seven
minutes for upload and verification after initial hashing. Future scheduling should
use representative checkpoint measurements and contingency, not extrapolate small
probes as a universal network rate. Intermediate architectures were not exhaustively
searched; **7.44B was the selected tested candidate, not a proven global maximum**.

## 9. Run Ledger and Deadline Management

| Job | Purpose | Nodes / GPUs | Outcome |
| --- | --- | --- | --- |
| `3209062` | Initial scheduled GPU check | 1 / 2 | Allocation granted; step blocked by inherited memory settings; released |
| `3209065` | Corrected two-GPU discovery | 1 / 2 | Passed |
| `3209069` | Eight-GPU discovery and topology | 1 / 8 | Passed |
| `3209119` | Hardware/software inventory | 1 / 8 | Completed; released |
| `3209342` | Multi-node device discovery | 2 / 16 | Passed; not yet a distributed training test |
| `3209586` | Maximum-without-delay readiness | 10 / 80 | Granted 32 CPUs and 128 GiB per node; private training stack was not ready; released |
| `3209817` | Full pinned-data preparation | 1 / 0 | Completed in 72 seconds |
| `3209888` | BF16 TP/DP correctness | 2 / 16 | Passed |
| `3209918` | 16.63B capacity probe | 1 / 8 | Probe completed; candidate rejected on safety/budget criteria |
| `3210021` | Real-data smoke training | 2 / 16 | Three updates and saved recovery state |
| `3210029` | 7.44B capacity probe | 1 / 8 | Passed one-node memory criterion |
| `3210061` | Main pilot | 12 / 96 | 21 updates; complete artifacts saved; FAILED/1 on distributed exit |
| `3210183` | Hardened shutdown diagnostic | 12 / 96 | Tiny model and DDP-gradient check; COMPLETED/0; no optimizer updates |

The main allocation requested 32 CPUs and 256 GiB host RAM per node, totaling
384 CPUs and 3 TiB RAM. It used a 60-second allocation-wait limit and an
application-level wall-clock training cutoff.

The provisional latest training cutoff was 15:15. After transfer measurements,
the pilot cutoff moved to **15:00**, reserving time for validation, checkpoint
staging, upload, and verification. The training loop considered estimated next-step
duration before starting another optimizer update.

| Time | Milestone |
| --- | --- |
| 14:15:27 | Tested contract/data foundation committed as `2212f88` |
| 14:39:40 | Model, training, checkpoint, and publisher implementation committed as `da7cd3f` |
| 14:49:38 | Selected pilot configuration committed as `bdadbe9` |
| 14:50:55 | Main Slurm job started |
| 14:52:38 | First optimizer update recorded |
| 14:59:32 | Last optimizer update recorded |
| About 15:03 | Final checkpoints and training summary completed |
| 15:03:10 | Main job ended with nonzero exit |
| 15:17:57 | Initial 33-file artifact set verified remotely |
| 15:29:28 | Hardened 12-node diagnostic completed cleanly |
| 15:48:47 | Refreshed 37-file payload set verified |
| 15:50:37 | Final pilot code and reports pushed to GitHub |
| 15:53:17 | Cross-remote checks confirmed GitHub, Hugging Face, and released GPU jobs |
| 16:00 | Original training reservation ended |

The 15:50 verification target was a planning target, not a falsely reported exact
completion time: the final cross-remote check was at 15:53:17, still before 16:00.
No model training was restarted after the cutoff. The later diagnostic was explicitly
a small computation/communication/lifecycle test without optimizer updates.

## 10. Training Results and Interpretation

| Fixed pilot-subset metric | Model | Constant-zero forecast |
| --- | ---: | ---: |
| Signed Huber | 5.023262 | 5.004593 |

Additional model metrics were MAE **5.490915** and RMSE **8.567953** log-return
percentage points. Prediction mean was approximately **-0.511945**, with standard
deviation **0.012205**; target mean was approximately **-0.242882**, with standard
deviation **8.563569**.

The predictions were nearly constant and did not beat the zero baseline. The run
covered only about **0.0096%** of one eligible training epoch. This cannot establish
the value of the architecture, the benefit of billions of parameters, convergence,
or a general forecasting advantage. It is also too early to diagnose a permanent
architectural collapse from this pilot alone.

Validation was a fixed market-stratified subset, not the complete validation anchor
population. Monitoring included an initial check and wall-clock opportunities in
addition to the planned sample threshold. The 50,000-example threshold was never
reached. Best selection was explicitly labelled as pilot-subset selection; the
test set remained untouched.

## 11. Failure Investigation and Its Limits

The main job's scheduler status and the checkpoint outcome were recorded separately.
The existence of a completion marker was not used to rewrite FAILED/1 as a clean
job exit. Best/latest pointers, all eight recovery shards, inference shards, counters,
and the summary were checked before publication.

The accessible terminal excerpt showed distributed-exit failures, but the full
client-side terminal log was not readable through the remote tooling. Therefore,
an exact root-cause claim would have exceeded the evidence.

The mitigation introduced explicit DP-group, TP-group, and world-group teardown,
worker exception recording, and persistent per-rank stdout/stderr. The diagnostic
was strengthened to retain the same DDP wrapper type used in training and compare
averaged gradients with a reference. It then completed cleanly on the 12-node mesh.

This validates the diagnostic and improved observability, **not a successful rerun
of the full model or a conclusive explanation of the original error**. That remaining
limit matters before scheduling a longer production run.

## 12. Checkpoint and Publication Workflow

Checkpoint directories were written by one DP replica's eight TP ranks, rather than
saving 12 redundant model replicas. Temporary files were completed before publication
markers and latest/best pointers were updated. Incomplete directories were rejected
by the publisher.

| Destination | Contents |
| --- | --- |
| [GitHub experiment folder](https://github.com/YLiu95/multi-step_forecast_MSc_project/tree/main/new%20experiment%201/new_experiment_1.5_asc_gpu_nodes) | Source, tests, launch/configuration files, dependency snapshot, plans, diary, and aggregate reports |
| [Hugging Face model repository](https://huggingface.co/YL95/experiment-1.5-asc-gpu-nodes) | One best inference checkpoint, one latest resumable checkpoint, model metadata/vocabulary, aggregate reports, and TensorBoard events |

Best and latest both referenced update 21 but served different purposes:

- **Best:** eight safetensors shards with BF16 inference matrices and FP32 replicated normalization parameters.
- **Latest:** eight PyTorch shards with full FP32 model/AdamW state, sampler/counter metadata, and the recorded TP/DP layout.

The first verified set contained 33 files and **104,331,123,677 bytes**. The final
payload set, after adding operational and reproducibility reports, contained
37 files and **104,331,130,121 bytes**. The later metadata refresh reused existing
checkpoint blobs rather than retransmitting the model data.

Verification compared each remote file's size and Git-blob or LFS digest against
the local artifact. The verification report was excluded from its own checksum
manifest and checked separately. GitHub's remote main commit was also compared
with the final local commit. Only the intended experiment directory was staged,
and the final working tree was clean.

The current repository tree contains one best/latest pair; Git/LFS history may
retain earlier revisions. No claim was made that history had been purged. Local
files were not deleted merely because an upload command returned successfully.

TensorBoard was served privately on the JupyterHub host at port **16007**. HTTP 200
and 153 scalar tags were verified. Users still need private port forwarding to view
that remote service. No public Cloudflare tunnel was created. The event file was
also verified in the Hugging Face artifact set.

## 13. Operational and Research Lessons

1. **Allocate resources through the scheduler, not through assumptions about the notebook host.** Kernel selection and remote GPU allocation are different operations.
2. **Keep scheduling forecasts separate from actual allocations.** Node count, CPU/RAM shape, duration, failures, and shared reservations all affect starts.
3. **Perform CPU-side preparation before holding many GPUs.** The full data preparation did not require GPU allocation.
4. **Validate TP math before scaling.** Device enumeration is not proof of compute, collectives, gradient equivalence, or correct sample accounting.
5. **Profile the complete distributed layout.** Full-DP reserved memory was higher than the one-node estimate; collect all-rank allocated/reserved peaks and reduce microbatch for longer runs.
6. **Plan backward from verified backup completion.** Budget measured checkpoint staging, representative upload duration, verification, and contingency before the reservation end.
7. **Persist rank logs before a long run.** A client terminal transcript is not a reliable substitute for durable worker error files.
8. **Keep research claims proportional to training evidence.** A large model and a successful checkpoint upload do not demonstrate forecasting skill.

The next research priority is a smaller matched-data baseline, a substantially
larger and explicitly budgeted training-example count, and robust full chronological
validation. Monitor prediction variance, calibration, country/ticker coverage, and
the zero-return baseline before considering further capacity increases. Retrospective
price adjustment, cleaning, and survivorship limitations still require a separate
point-in-time data audit before any trading interpretation.

Exact continuation through the current resume path requires the recorded model/data
configuration and **TP=8/DP=12** layout. A smaller overnight allocation is not an
automatically supported elastic-DP resume; that change needs explicit implementation
and validation of optimizer, sampling, and global-batch semantics.

## 14. Post-Pilot Reservation Transition

This is a separate operational postscript, not an extension of the pilot deadline.
Read-only checks at **16:18 on 2026-09-29** showed:

| Reservation | Window | State and reserved nodes |
| --- | --- | --- |
| `training` | 2026-09-29 16:00 to 2026-09-30 08:00 | Active; `t[0001-0004]`, four nodes, with FLEX |
| `training-multigpu-2` | 2026-09-30 09:00 to 16:00 | Scheduled but inactive at the check; 15 nodes |

Separate earlier post-pilot dry runs had forecast opportunistic allocations beyond
the four reserved nodes. They did not hold those resources and must not be treated
as a permanent guarantee. The GPU reservation does not extend the JupyterHub or
VS Code tunnel job's lifetime; check that job's end time independently and use
batch/checkpoint workflows for work that must survive a disconnected session.

No new GPU job or model training was launched to prepare this report. The original
pilot's checkpoints and code were already backed up before its 16:00 cutoff.