"""Generates 'resume training.ipynb'. Kept as a script so the notebook JSON is
never hand-edited (and so it can be regenerated after a config change)."""
import json
from pathlib import Path

C = []


def md(text):
    C.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    C.append({"cell_type": "code", "execution_count": None, "metadata": {},
              "outputs": [], "source": text.strip("\n").splitlines(True)})


md(r"""
# Resume Training

Continue a run from **any** checkpoint — the one that was interrupted, an older epoch you want
to branch from, or a checkpoint pulled back down from Hugging Face after the environment died.

Everything you need to change is in the **single configuration cell** below. Nothing else in
this notebook needs editing.

## What "resume" actually has to restore

A common bug is to reload only `model.state_dict()`. Restart from that and Adam's first and
second moment estimates are gone: the first few hundred steps take large, badly-scaled steps
and can undo hours of training. The learning-rate schedule also restarts from step 0, so you
get a fresh warmup in the middle of a cosine decay.

The full checkpoints written by `CheckpointCallback` therefore contain:

| Key | Why it must be saved |
|---|---|
| `model` | the weights |
| `optimizer` | **Adam's momentum + variance** — the thing everyone forgets |
| `scaler` | the fp16 loss scale, so AMP does not have to re-discover it |
| `ema` | the EMA shadow weights used for evaluation |
| `global_step` | so the cosine LR schedule continues instead of restarting |
| `best_value` | so early stopping and "is this a new best?" keep their memory |
| `rng` | torch / cuda / numpy RNG state |
| `config` | the exact config the checkpoint was trained with |

That is why a full checkpoint is ~612 MB while `best.pt` (weights only, for inference) is
~306 MB.

## Contents
| # | Section |
|---|---------|
| 1 | [Configuration](#c1) — the only cell you edit |
| 2 | [Find available checkpoints](#c2) (local **and** Hugging Face) |
| 3 | [Inspect the checkpoint](#c3) before you commit to it |
| 4 | [Open TensorBoard](#c4) |
| 5 | [Visualise current and past runs](#c5) |
| 6 | [Resume](#c6) |
| 7 | [Monitor](#c7) |
| 8 | [Troubleshooting](#c8) |
""")

md(r"""
<a id="c1"></a>
## 1. Configuration — the only cell you need to edit
""")

code(r'''
# =============================================================================
#  CONFIGURATION  -  the only cell you need to edit
# =============================================================================

PROJECT_DIR   = "/root/new_experiment_1/new experiment 1"   # git-backed source
ARTIFACT_ROOT = "/root/artifacts"                           # data, logs, checkpoints

# ---- 1. WHICH RUN TO RESUME -------------------------------------------------
RUN_NAME      = "patchtst_us_equities_v1"
                              # The run to continue. Its logs live in
                              # <ARTIFACT_ROOT>/runs/<RUN_NAME>/ and its
                              # checkpoints in <ARTIFACT_ROOT>/checkpoints/<RUN_NAME>/.

# ---- 2. WHICH CHECKPOINT ----------------------------------------------------
CHECKPOINT    = "auto"
                              # "auto"    -> local latest.pt (the normal case)
                              # "best"    -> local best.pt (weights only: restarts
                              #              the optimiser, so only branch from it
                              #              deliberately)
                              # "hf:latest.pt"            -> pull from Hugging Face
                              # "hf:ckpt_epoch_025.pt"    -> a specific HF epoch
                              # "/abs/path/to/ckpt.pt"    -> any file

# ---- 3. HOW MUCH FURTHER TO TRAIN -------------------------------------------
EXTRA_EPOCHS  = 0             # 0    -> run to the original `epochs` target
                              # n>0  -> extend the target by n epochs.
                              # NOTE: extending changes the cosine schedule, which
                              # is annealing towards the ORIGINAL end epoch. A run
                              # that already annealed to min_lr will barely move
                              # unless you also raise NEW_LR.

# ---- 4. OPTIONAL OVERRIDES --------------------------------------------------
# Leave as None to keep whatever the checkpoint was trained with.
NEW_LR              = None    # e.g. 1e-4 to fine-tune at a lower rate
NEW_BATCH_SIZE      = None    # e.g. 512 if you resume on a smaller GPU
NEW_STEPS_PER_EPOCH = None    # e.g. 250 for faster feedback
NEW_RUN_NAME        = None    # set to fork into a NEW run instead of continuing
                              # this one (keeps the old TensorBoard curves intact)

# ---- 5. BACKUPS -------------------------------------------------------------
PUSH_TO_HF     = True         # upload checkpoints to YL95/new_experiment_1
PUSH_TO_GITHUB = True         # commit code + history to the project repo

# ---- 6. TENSORBOARD ---------------------------------------------------------
TB_PORT        = 6006
TB_LOGDIR      = f"{ARTIFACT_ROOT}/runs"   # parent dir -> shows ALL runs together

# =============================================================================

import sys, os, json, subprocess, time
from pathlib import Path

sys.path.insert(0, PROJECT_DIR)
from src.config import Config
from src.hub import HFBackup

cfg = Config().override(run_name=RUN_NAME, artifact_root=ARTIFACT_ROOT)
CKPT_DIR = cfg.paths["ckpt"]
RUNS_DIR = Path(ARTIFACT_ROOT) / "runs"

print(f"project      {PROJECT_DIR}")
print(f"run          {RUN_NAME}")
print(f"checkpoints  {CKPT_DIR}")
print(f"tb logdir    {TB_LOGDIR}")
print(f"requested    CHECKPOINT={CHECKPOINT!r}  EXTRA_EPOCHS={EXTRA_EPOCHS}")
''')

md(r"""
<a id="c2"></a>
## 2. Find available checkpoints

Looks in two places: the local checkpoint directory, and the Hugging Face repo. The second one
is what you need after the environment has been wiped — the local disk is gone, but
`checkpoints/` on the Hub is not.
""")

code(r'''
print("=" * 92); print(f" LOCAL  {CKPT_DIR}"); print("=" * 92)
local = sorted(CKPT_DIR.glob("*.pt")) if CKPT_DIR.exists() else []
if not local:
    print("  (none - the local disk may have been wiped; check Hugging Face below)")
for p in local:
    print(f"  {p.name:<28} {p.stat().st_size / 1e6:>7,.0f} MB   "
          f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))}")

print("\n" + "=" * 92); print(f" HUGGING FACE  {cfg.hf_repo_id}"); print("=" * 92)
hf = HFBackup(cfg.hf_repo_id, enabled=True)
remote = [f for f in hf.list_files() if f.endswith((".pt", ".jsonl"))]
if not remote:
    print("  (nothing found, or no HF token available)")
for f in remote:
    print(f"  {f}")

print("\n" + "=" * 92); print(" RUNS WITH LOGS"); print("=" * 92)
for d in sorted(RUNS_DIR.glob("*")) if RUNS_DIR.exists() else []:
    h = d / "history.jsonl"
    n = sum(1 for _ in open(h)) if h.exists() else 0
    print(f"  {d.name:<34} {n:>4} epochs logged")
''')

md(r"""
<a id="c3"></a>
## 3. Resolve and inspect the checkpoint

Never resume blind. This cell loads the checkpoint **onto the CPU** (so it cannot disturb a
run already using the GPUs), prints what is inside it, and warns you if anything is missing.
""")

code(r'''
import torch

def resolve_checkpoint(spec: str) -> Path | None:
    if spec == "auto":
        p = CKPT_DIR / "latest.pt"
        return p if p.exists() else None
    if spec == "best":
        p = CKPT_DIR / "best.pt"
        return p if p.exists() else None
    if spec.startswith("hf:"):
        name = spec[3:]
        for prefix in (cfg.hf_ckpt_dir, cfg.hf_best_dir):
            got = hf.download(f"{prefix}/{name}", CKPT_DIR)
            if got:
                print(f"downloaded {prefix}/{name} -> {got}")
                return got
        return None
    p = Path(spec)
    return p if p.exists() else None

ckpt_path = resolve_checkpoint(CHECKPOINT)
if ckpt_path is None:
    print(f"NO CHECKPOINT FOUND for {CHECKPOINT!r}.")
    print("Training will start from scratch if you run section 6 anyway.")
else:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print("=" * 92); print(f" {ckpt_path}"); print("=" * 92)
    print(f"  epoch           {ck.get('epoch')}")
    print(f"  global_step     {ck.get('global_step'):,}")
    print(f"  best_value      {ck.get('best_value')}")
    print(f"  n_features      {ck.get('n_features')}")
    print(f"  torch version   {ck.get('torch_version')}   (now running {torch.__version__})")
    print(f"  file size       {ckpt_path.stat().st_size / 1e6:,.0f} MB")

    print("\n  contents:")
    for key in ("model", "optimizer", "scaler", "ema", "rng", "config"):
        mark = "yes" if key in ck else "MISSING"
        print(f"    {key:<12} {mark}")
    if "optimizer" not in ck:
        print("\n  WARNING: no optimizer state. Adam's moments will be re-estimated from")
        print("           zero, so expect a bump in the loss for the first ~200 steps.")

    saved = Config.from_dict(ck["config"])
    diffs = {k: (v, getattr(cfg, k)) for k, v in saved.to_dict().items()
             if getattr(cfg, k, None) != v and k not in ("feature_names",)}
    print("\n  config differences (checkpoint -> current):")
    if not diffs:
        print("    none")
    for k, (a, b) in diffs.items():
        print(f"    {k:<22} {a}  ->  {b}")
    # Anything that changes tensor shapes makes the weights unloadable.
    breaking = {"d_model", "depth", "n_heads", "d_ff", "patch_len", "patch_stride",
                "n_steps_in", "n_steps_out", "loss", "quantiles"}
    bad = breaking & set(diffs)
    if bad:
        print(f"\n  ERROR: {sorted(bad)} change the model's SHAPE. The weights will not")
        print("         load. Either revert them or start a new run.")
    del ck
''')

md(r"""
<a id="c4"></a>
## 4. Open TensorBoard

`TB_LOGDIR` points at the **parent** `runs/` directory, so TensorBoard shows every run — the
one you are resuming *and* every past one — on the same axes.

### Option A — inline (run the next cell)

### Option B — VS Code's built-in panel
1. **`Ctrl+Shift+P`** (`Cmd+Shift+P` on macOS)
2. Type **`Python: Launch TensorBoard`**
3. Choose **"Select another folder"** → `/root/artifacts/runs`

### Option C — browser over the tunnel
```bash
tensorboard --logdir /root/artifacts/runs --port 6006 --bind_all
```
Then open the **PORTS** panel in VS Code and click the 🌐 globe icon next to port `6006`.

> **A resumed run continues the same curves.** `global_step` is restored from the checkpoint,
> so the new points land after the old ones instead of overwriting them from step 0. If you set
> `NEW_RUN_NAME`, you get a separate curve to compare against instead.
""")

code(r'''
%load_ext tensorboard
%tensorboard --logdir /root/artifacts/runs --port 6006
''')

md(r"""
<a id="c5"></a>
## 5. Visualise current and past runs

Reads every `runs/*/history.jsonl` and overlays them. This works even while training is
writing, because the history file is append-only JSON lines.
""")

code(r'''
import pandas as pd
import matplotlib.pyplot as plt

def load_histories(runs_dir=RUNS_DIR) -> dict[str, pd.DataFrame]:
    out = {}
    for d in sorted(Path(runs_dir).glob("*")):
        h = d / "history.jsonl"
        if not h.exists():
            continue
        try:
            df = pd.read_json(h, lines=True)
        except ValueError:      # a partially written last line while training
            lines = [l for l in h.read_text().splitlines() if l.strip()]
            df = pd.DataFrame([json.loads(l) for l in lines[:-1]])
        if len(df):
            out[d.name] = df
    return out

hists = load_histories()
if not hists:
    print("No run history yet. The first epoch appears after ~6 minutes of training.")
else:
    # ---------------------------------------------------------- summary table
    rows = []
    for name, df in hists.items():
        b = df.loc[df["val_loss"].idxmin()]
        rows.append({
            "run": name, "epochs": len(df),
            "best_epoch": int(b["epoch"]), "best_val_loss": b["val_loss"],
            "r2": b.get("val_r2_vs_zero", float("nan")),
            "rank_ic": b.get("val_rank_ic", float("nan")),
            "dir_acc": b.get("val_dir_acc", float("nan")),
            "std_ratio": b.get("val_std_ratio", float("nan")),
            "min/epoch": df["epoch_seconds"].mean() / 60,
        })
    print(pd.DataFrame(rows).sort_values("best_val_loss")
            .to_string(index=False, float_format=lambda v: f"{v:,.5f}"))

    # ------------------------------------------------------------- the panels
    panels = [("val_loss",       "validation loss",                 None),
              ("train_loss",     "training loss",                   None),
              ("val_r2_vs_zero", "val $R^2$ vs zero  (+0.005..0.02 = good)", 0.0),
              ("val_rank_ic",    "val rank IC  (0.02..0.05 = real signal)",  0.0),
              ("val_dir_acc",    "val directional accuracy",        0.5),
              ("val_std_ratio",  "val std_ratio  ($\\to$ 0 = model gave up)", None)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    for ax, (col, title, ref) in zip(axes.ravel(), panels):
        for name, df in hists.items():
            if col in df:
                ax.plot(df["epoch"], df[col], label=name, lw=1.6)
        if ref is not None:
            ax.axhline(ref, color="k", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("All runs, current and past", y=1.00)
    plt.tight_layout(); plt.show()

    # -------------------------------------------- where the current run stands
    if RUN_NAME in hists:
        df = hists[RUN_NAME]
        last = df.iloc[-1]
        print(f"\n{RUN_NAME}: {len(df)} epochs done of {cfg.epochs}")
        print(f"  last  epoch {int(last['epoch'])}  val_loss {last['val_loss']:.6f}")
        eta = df["epoch_seconds"].tail(5).mean() * (cfg.epochs - len(df)) / 60
        print(f"  ETA to target: ~{eta:,.0f} min")
''')

md(r"""
<a id="c6"></a>
## 6. Resume

Launches `torchrun` with `--resume`, detached from this kernel so the run survives a notebook
restart.

> **`torchrun`, not a call in this cell.** DDP needs one OS process per GPU and a notebook
> kernel is a single process. `nn.DataParallel` would run in-process but the GIL serialises the
> forward passes and gradients pile up on GPU 0 — expect ~1.3× instead of ~1.95×.
""")

code(r'''
overrides = []
if NEW_LR is not None:
    overrides += ["--set", f"lr={NEW_LR}"]
if NEW_BATCH_SIZE is not None:
    overrides += ["--set", f"batch_size={NEW_BATCH_SIZE}"]
if NEW_STEPS_PER_EPOCH is not None:
    overrides += ["--set", f"steps_per_epoch={NEW_STEPS_PER_EPOCH}"]
if EXTRA_EPOCHS:
    overrides += ["--set", f"epochs={cfg.epochs + EXTRA_EPOCHS}"]

target_run = NEW_RUN_NAME or RUN_NAME
resume_arg = str(ckpt_path) if ckpt_path else "auto"

cmd = ["torchrun", "--nproc_per_node=2", "-m", "src.train",
       "--run-name", target_run, "--resume", resume_arg, *overrides]
if not PUSH_TO_HF:
    cmd.append("--no-hf")
if not PUSH_TO_GITHUB:
    cmd.append("--no-git")

RESUME_LOG = f"{ARTIFACT_ROOT}/resume_{target_run}.log"
already = subprocess.run(["pgrep", "-f", "src.train"], capture_output=True).returncode == 0

print("$", " ".join(f"'{c}'" if " " in c else c for c in cmd), "\n")
if already:
    print("Training is ALREADY running - not launching a second copy.")
    print("Two runs would fight over the same GPUs and both would be slower.")
    print("To stop the current one:  !pkill -f src.train")
else:
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with open(RESUME_LOG, "w") as log:
        subprocess.Popen(cmd, cwd=PROJECT_DIR, env=env, stdout=log,
                         stderr=subprocess.STDOUT, start_new_session=True)
    print(f"launched. log -> {RESUME_LOG}")
    time.sleep(25)
    print("\n--- first lines ---")
    print(subprocess.run(["tail", "-n", "6", RESUME_LOG],
                         capture_output=True, text=True).stdout)
''')

md(r"""
<a id="c7"></a>
## 7. Monitor

Re-run freely — it only reads files.
""")

code(r'''
LOG = RESUME_LOG if Path(RESUME_LOG).exists() else f"{ARTIFACT_ROOT}/train.log"

print("=" * 92); print(" GPUs"); print("=" * 92)
print(subprocess.run(
    ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
     "--format=csv,noheader"], capture_output=True, text=True).stdout)
print(f"training running: "
      f"{subprocess.run(['pgrep', '-f', 'src.train'], capture_output=True).returncode == 0}")

hist_path = Path(ARTIFACT_ROOT) / "runs" / (NEW_RUN_NAME or RUN_NAME) / "history.jsonl"
print("\n" + "=" * 92); print(f" HISTORY  {hist_path}"); print("=" * 92)
if hist_path.exists():
    df = pd.read_json(hist_path, lines=True)
    cols = [c for c in ["epoch", "train_loss", "val_loss", "val_r2_vs_zero",
                        "val_rank_ic", "val_dir_acc", "val_std_ratio", "lr",
                        "epoch_seconds"] if c in df.columns]
    print(df[cols].tail(15).to_string(index=False, float_format=lambda v: f"{v:,.5f}"))
else:
    print("  no epochs finished yet")

print("\n" + "=" * 92); print(f" LOG TAIL  {LOG}"); print("=" * 92)
print(subprocess.run(["tail", "-n", "15", LOG], capture_output=True, text=True).stdout)
''')

md(r"""
<a id="c8"></a>
## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` right after launch | An old run still holds VRAM | `!pkill -f src.train`, wait 10 s, check `nvidia-smi`, relaunch |
| `size mismatch for ...` when loading | A shape-changing config field differs (`d_model`, `depth`, `n_steps_in`, `loss`, …) | Section 3 lists exactly which. Revert it, or set `NEW_RUN_NAME` and train fresh |
| Loss jumps up right after resuming | The checkpoint had no optimiser state (e.g. you resumed from `best.pt`) | Resume from `latest.pt` or a `ckpt_epoch_*.pt` instead |
| Run hangs with both GPUs at 0% | A DDP deadlock — one rank exited and the others wait on an all-reduce | `pkill -f src.train` and resume; `src/train.py` broadcasts the stop flag to prevent this |
| `address already in use` | A previous `torchrun` did not release its port | `torchrun --rdzv-endpoint=localhost:29501 ...` |
| `TensorBoard could not bind to port 6006` | An instance is already running (probably serving what you want) | `curl -s localhost:6006/data/runs` to check it, or `pkill -f "tensorboard.*6006"` and relaunch, or use `--port 6007` |
| TensorBoard shows "No dashboards are active" | Pointed at a run dir instead of the parent, or the log dir was deleted/recreated after TensorBoard started | Use `/root/artifacts/runs`; if the path is right, restart TensorBoard |
| Nothing in TensorBoard | Pointed at a run directory instead of the parent | Use `/root/artifacts/runs`, not `/root/artifacts/runs/<name>/tb` |
| Local checkpoints gone after a restart | The environment was wiped | Use `CHECKPOINT = "hf:latest.pt"` to pull from Hugging Face |
| LR barely moves after extending epochs | Cosine had already annealed to `min_lr` | Also set `NEW_LR` |

### Restoring after the environment is wiped

```python
CHECKPOINT = "hf:latest.pt"     # in the configuration cell
```

Then run sections 2 → 3 → 6. You will also need the data panel; rebuild it with

```bash
cd "/root/new_experiment_1/new experiment 1" && python -m src.prepare_data
```

which takes ~9 minutes from a cold start (the parquet download chunks are cached, so if
`/root/artifacts/cache` survived it is only seconds).
""")

nb = {
    "cells": C,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.13",
                          "mimetype": "text/x-python",
                          "codemirror_mode": {"name": "ipython", "version": 3},
                          "file_extension": ".py", "pygments_lexer": "ipython3",
                          "nbconvert_exporter": "python"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out = Path("/root/resume training.ipynb")
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(C)} cells)")
