#!/usr/bin/env bash
set -euo pipefail

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

# ===== User-configurable =====
# If training already saves HF safetensors, you can skip this step.
# Otherwise export Megatron (torch_dist) to HF safetensors.
MCORE_CKPT_DIR="/data/finetuned_layer61"
HF_SAVE_DIR="/data/finetuned_layer61-hf"

# Config to match 16-GPU Training
# Export can usually run on fewer GPUs if memory allows, but loading the checkpoint
# specifices the structure.
TP=16
PP=1
EP=1

# Run on 16 GPUs is usually enough for export if simply converting
NPROC_PER_NODE=16
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NPROC_PER_NODE=${NPROC_PER_NODE} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
megatron export \
  --load "${MCORE_CKPT_DIR}" \
  --save "${HF_SAVE_DIR}" \
  --to_hf true \
  --tensor_model_parallel_size ${TP} \
  --pipeline_model_parallel_size ${PP} \
  --expert_model_parallel_size ${EP} \
  --test_convert_precision false
