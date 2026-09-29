#!/usr/bin/env bash
set -euo pipefail
cd "${EXPERIMENT_DIR:-$SLURM_SUBMIT_DIR}"
source ./env.sh
master_host="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
master_address="$(getent ahostsv4 "$master_host" | awk 'NR == 1 { print $1 }')"
export MASTER_ADDR="$master_address"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
unset GITHUB_TOKEN GH_TOKEN HF_TOKEN HUGGING_FACE_HUB_TOKEN
exec python -m torch.distributed.run \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc_per_node=8 \
    --node_rank="$SLURM_PROCID" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --max_restarts=0 \
    -m "$@"