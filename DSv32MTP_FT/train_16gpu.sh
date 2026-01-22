#!/usr/bin/env bash
set -euo pipefail

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

# ===== User-configurable =====
MODEL_DIR="/data/models/DeepSeek-V3.2"
DATASET_PATH="${SCRIPT_DIR}/sft_10.jsonl"
OUTPUT_DIR="/data/finetuned_layer61"

# Parallelism (16 GPUs Required)
# Configured for 2 nodes x 8 GPUs (Standard H200 setup)
# or 1 node x 16 GPUs.
# Total World Size must be 16.

# Default to running on a single node with 16 GPUs if not specified? 
# Usually H200 servers have 8. So this likely requires multi-node.
# The user needs to run this script on each node with appropriate env vars
# (NODE_RANK, MASTER_ADDR, etc.) or use a launcher.
# Here we set the model parallelism parameters to consume 16 GPUs.

TP=16
PP=1
EP=1
CP=1

# Check if 16 GPUs are available in total provided by environment
# (This is just a script config; runtime check is done by torchrun/megatron)

# Training hyperparams
MICRO_BATCH=1
GLOBAL_BATCH=128
EPOCHS=1
LR=2e-5
MIN_LR=2e-6
MAX_LENGTH=4096
EVAL_INTERVAL=500
SAVE_INTERVAL=500

# FP8 Settings (Required: Layer 61 output as block quant fp8)
FP8_FORMAT="e4m3"
FP8_RECIPE="blockwise"
FP8_ARGS=(
    --fp8_format "${FP8_FORMAT}"
    --fp8_recipe "${FP8_RECIPE}"
    --use_precision_aware_optimizer true
    --exp_avg_dtype fp8
    --exp_avg_sq_dtype fp8
)

# Trainable regex: Layer 60 (last layer) + Heads + Norms
TRAINABLE_REGEX="model\.layers\.60\.|lm_head|model\.norm|model\.final_layernorm"

# ===== Train (Megatron-SWIFT) =====
# Using TP=16, PP=1 to split model across 16 GPUs
# PP Stage 1: Layers 0-30 (Frozen)
# PP Stage 2: Layers 31-60 (Frozen except Layer 60)

echo "Starting DeepSeek-V3.2 16-GPU Fine-tuning..."
echo "TP=${TP}, PP=${PP} => Total GPUs: $((TP*PP))"

# Note: NPROC_PER_NODE should be set by the caller or 00_env.sh. 
# If running single node 16 gpu: NPROC_PER_NODE=16
# If running 2 nodes 8 gpu: NPROC_PER_NODE=8 (default in 00_env.sh)

megatron sft \
  --model "${MODEL_DIR}" \
  --model_type deepseek_v3_2 \
  --load_safetensors true \
  --save_safetensors true \
  --train_type full \
  --finetune true \
  --dataset "${DATASET_PATH}" \
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
  "${FP8_ARGS[@]}"
