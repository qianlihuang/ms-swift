#!/usr/bin/env bash
# ============================================================================
# DeepSeek-V3.2 MTP Last Layer Fine-tuning (Quick Test)
# 
# 目标: 快速验证训练流程，仅使用10条数据
# ============================================================================
set -euo pipefail

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

# ===== Configuration =====
MODEL_DIR="/data/models/DeepSeek-V3.2"
DATASET_PATH="/workspace/ms-swift/DSv32MTP_FT/sft_10.jsonl"
OUTPUT_DIR="/data/finetuned_layer60_test"

# Parallelism: 8x H200, TP=1, EP=8
# 每个GPU加载 256/8=32 个experts
NPROC_PER_NODE=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TP=1
PP=1
CP=1
EP=8

# Training hyperparams (minimal for quick test)
MICRO_BATCH=1
GLOBAL_BATCH=8
EPOCHS=1
LR=2e-5
MIN_LR=2e-6
MAX_LENGTH=1024

# 只训练最后一层的部分参数 + lm_head + norm
# 避免训练所有256个experts以减少显存
TRAINABLE_REGEX="model\.layers\.60\.(self_attn|input_layernorm|post_attention_layernorm|mlp\.gate|mlp\.shared_experts)|lm_head|model\.norm"

# ===== Check =====
echo "============================================"
echo "DeepSeek-V3.2 MTP Fine-tuning (Quick Test)"
echo "============================================"
echo "Model: ${MODEL_DIR}"
echo "Dataset: ${DATASET_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Parallelism: TP=${TP}, EP=${EP}"
echo "============================================"

if [[ ! -f "${DATASET_PATH}" ]]; then
    echo "[ERROR] Dataset not found: ${DATASET_PATH}"
    exit 1
fi

# ===== Train =====
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NPROC_PER_NODE=${NPROC_PER_NODE} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
megatron sft \
    --model "${MODEL_DIR}" \
    --model_type deepseek_v3_2 \
    --dataset "${DATASET_PATH}" \
    --train_type full \
    --freeze_parameters_ratio 1 \
    --trainable_parameters_regex "${TRAINABLE_REGEX}" \
    --tensor_model_parallel_size ${TP} \
    --pipeline_model_parallel_size ${PP} \
    --context_parallel_size ${CP} \
    --expert_model_parallel_size ${EP} \
    --moe_grouped_gemm true \
    --moe_permute_fusion true \
    --moe_shared_expert_overlap false \
    --moe_aux_loss_coeff 0 \
    --micro_batch_size ${MICRO_BATCH} \
    --global_batch_size ${GLOBAL_BATCH} \
    --recompute_granularity selective \
    --max_epochs ${EPOCHS} \
    --lr ${LR} \
    --lr_warmup_fraction 0.1 \
    --min_lr ${MIN_LR} \
    --save "${OUTPUT_DIR}" \
    --save_interval 50 \
    --max_length ${MAX_LENGTH} \
    --num_workers 4 \
    --dataset_num_proc 4 \
    --no_save_optim true \
    --no_save_rng true

echo ""
echo "Training completed! Output: ${OUTPUT_DIR}"
