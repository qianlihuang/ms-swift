#!/usr/bin/env bash
set -euo pipefail

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===== User-configurable =====
MODEL_DIR="/data/models/DeepSeek-V3.2"
DATASET_PATH="${SCRIPT_DIR}/sft_10.jsonl"
OUTPUT_DIR="/data/finetuned_layer61"

# Parallelism (16 GPUs Required)
# Configured for 2 nodes x 8 GPUs (Standard H200 setup) -> Total 16 GPUs
# If using TP=16, you MUST have NVLink across all 16 GPUs (e.g. DGX GH200) or acceptable interconnect performance.
# Standard H200 nodes usually have 8 GPUs. TP=16 might be slow across nodes without high-bandwidth (NVSwitch) cross-node.
# IF 2 nodes, set NNODES=2, NODE_RANK=0 (on node 1) and NODE_RANK=1 (on node 2).

TP=16
PP=1
EP=1
CP=1

# Distributed Environment Variables
# Defaults for a 2-node setup (since 8 GPUs per node is standard)
# Run this script on EACH node with appropriate NODE_RANK
export NNODES=${NNODES:-2}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29500}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# Construct GLOO/NCCL options if needed, but torchrun/megatron usually handles it via env vars.
# We explicitly export them above.

# Check if we should warn about TP=16 on multi-node without NVSwitch
echo "Starting DeepSeek-V3.2 16-GPU Fine-tuning..."
echo "Parallelism: TP=${TP}, PP=${PP}, EP=${EP}"
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

# FP8 Settings (Required: Layer 61 output as block quant fp8)
FP8_FORMAT="e4m3"
# 'blockwise' (per-block scaling) or 'tensorwise' (per-tensor scaling)
# DeepSeek V3 typically uses fine-grained quantization.
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

# ===== Train (Megatron-SWIFT) =====
# Using TP=16, PP=1 to split model across 16 GPUs
# PP Stage 1: Layers 0-60 (Full model in 1 stage, just split tensor-wise)

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
  "${FP8_ARGS[@]}"
