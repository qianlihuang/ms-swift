#!/usr/bin/env bash
set -euo pipefail

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

# ===== User-configurable =====
MODEL_DIR="/data/models/DeepSeek-V3.2"
DATASET_PATH="/data/datasets/sft.jsonl"
OUTPUT_DIR="/data/finetuned_layer61"

# Parallelism (single-node 8xH200 by default)
NPROC_PER_NODE=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TP=1
PP=1
CP=1
EP=8

# Training hyperparams
MICRO_BATCH=1
GLOBAL_BATCH=8
EPOCHS=1
LR=2e-5
MIN_LR=2e-6
MAX_LENGTH=4096
EVAL_INTERVAL=500
SAVE_INTERVAL=500

# MTP: set to 1 only if MTP head exists (see 10_inspect_params.py)
MTP_NUM_LAYERS=0

# Trainable regex: update based on 10_inspect_params.py output
TRAINABLE_REGEX="model\.layers\.60\.|lm_head|model\.norm|model\.final_layernorm"

# ===== MTP args (optional) =====
MTP_ARGS=()
if [[ "${MTP_NUM_LAYERS}" -gt 0 ]]; then
  MTP_ARGS=(--mtp_num_layers "${MTP_NUM_LAYERS}")
fi

# ===== FP8 settings =====
# Requires: TransformerEngine + CUDA 12.9+ for blockwise fp8, megatron-core>=0.15.
FP8_FORMAT="e4m3"
FP8_RECIPE="blockwise"

# ===== Train (Megatron-SWIFT + FP8) =====
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NPROC_PER_NODE=${NPROC_PER_NODE} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
megatron sft \
  --model "${MODEL_DIR}" \
  --model_type deepseek_v3_2 \
  --load_safetensors true \
  --save_safetensors true \
  --train_type full \
  --finetune true \
  --dataset "${DATASET_PATH}" \
  "${MTP_ARGS[@]}" \
  --freeze_parameters_ratio 1 \
  --trainable_parameters_regex "${TRAINABLE_REGEX}" \
  --tensor_model_parallel_size ${TP} \
  --pipeline_model_parallel_size ${PP} \
  --context_parallel_size ${CP} \
  --expert_model_parallel_size ${EP} \
  --moe_grouped_gemm true \
  --moe_permute_fusion true \
  --moe_shared_expert_overlap true \
  --moe_aux_loss_coeff 1e-6 \
  --micro_batch_size ${MICRO_BATCH} \
  --global_batch_size ${GLOBAL_BATCH} \
  --recompute_granularity full \
  --recompute_method uniform \
  --recompute_num_layers 1 \
  --max_epochs ${EPOCHS} \
  --lr ${LR} \
  --lr_warmup_fraction 0.05 \
  --min_lr ${MIN_LR} \
  --save "${OUTPUT_DIR}" \
  --eval_interval ${EVAL_INTERVAL} \
  --save_interval ${SAVE_INTERVAL} \
  --max_length ${MAX_LENGTH} \
  --num_workers 8 \
  --dataset_num_proc 8 \
  --no_save_optim true \
  --no_save_rng true \
  --sequence_parallel true \
  --attention_backend flash \
  --fp8_format ${FP8_FORMAT} \
  --fp8_recipe ${FP8_RECIPE} \
  --use_precision_aware_optimizer true \
  --exp_avg_dtype fp8 \
  --exp_avg_sq_dtype fp8
