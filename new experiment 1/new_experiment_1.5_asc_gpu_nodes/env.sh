#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.venvs/experiment-1.5-gpu/bin:$PATH"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-/net/tscratch/people/$(id -un)/experiments/experiment_1.5_asc_gpu_nodes}"
IFS=: read -r -a library_paths <<< "${LD_LIBRARY_PATH:-}"
filtered_library_path=""
for library_path in "${library_paths[@]}"; do
    case "$library_path" in
        */CUDA/*|*/NCCL/*|*/NVHPC/*|*/UCX-CUDA/*|*/UCC-CUDA/*|*/GDRCopy/*) continue ;;
    esac
    if [[ -n "$library_path" ]]; then
        filtered_library_path="${filtered_library_path:+$filtered_library_path:}$library_path"
    fi
done
export LD_LIBRARY_PATH="$filtered_library_path"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1