#!/usr/bin/env bash
set -euo pipefail

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===== User-configurable =====
MODEL_DIR="/data/models/DeepSeek-V3.2"
DATASET_PATH="${SCRIPT_DIR}/sft_10.jsonl"
OUTPUT_DIR="/data/finetuned_layer61_tp8_pp3"

# Parallelism (24 GPUs Required)
# Configured for 3 nodes x 8 GPUs = 24 GPUs
# TP=8 (fit in one node), PP=3 (across 3 nodes)
# Run this script on EACH node with appropriate NODE_RANK ranges (0, 1, 2)

TP=8
PP=3
EP=1
CP=1

# Distributed Environment Variables
# Defaults for a 3-node setup
export NNODES=${NNODES:-3}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29500}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}

echo "Starting DeepSeek-V3.2 24-GPU Fine-tuning (TP=${TP}, PP=${PP})..."
echo "Distributed: Nodes=${NNODES}, Rank=${NODE_RANK}, Master=${MASTER_ADDR}:${MASTER_PORT}"

# Training hyperparams
MICRO_BATCH=1
GLOBAL_BATCH=128
EPOCHS=1
LR=2e-5
MIN_LR=2e-6
MAX_LENGTH=4096
EVAL_INTERVAL=500
SAVE_INTERVAL=500

# FP8 Settings
FP8_FORMAT="e4m3"
FP8_RECIPE="blockwise"

FP8_ARGS=(
    --fp8_format "${FP8_FORMAT}"
    --fp8_recipe "${FP8_RECIPE}"
    --use_precision_aware_optimizer true
    --exp_avg_dtype fp8
    --exp_avg_sq_dtype fp8
)

# Trainable regex: Layer 61 (last layer) + Heads + Norms
TRAINABLE_REGEX="model\.layers\.61\.|lm_head|model\.norm"

# Pipeline Config for 61 Layers
# Total 61 Layers. PP=3.
# Split suggestion: 20 (First) + 21 (Middle) + 20 (Last) = 61
# First and Last stages have Embedding/Head overhead, so we give them 1 less layer than middle.
PP_ARGS=(
    --decoder_first_pipeline_num_layers 20
    --decoder_last_pipeline_num_layers 20
)

# ===== Train (Megatron-SWIFT) =====

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
  --mtp_num_layers 1 \
  "${FP8_ARGS[@]}" \
  "${PP_ARGS[@]}"
