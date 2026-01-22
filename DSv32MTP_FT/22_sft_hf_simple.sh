#!/usr/bin/env bash
# ============================================================================
# DeepSeek-V3.2 MTP Last Layer Fine-tuning Script
# 
# 目标: 微调最后一层 (layer 60) 和 lm_head 以提高投机解码接受率
# 使用: ms-swift 的 swift sft 命令 (HuggingFace 后端)
# ============================================================================
set -euo pipefail

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

# ===== User-configurable =====
MODEL_DIR="/data/models/DeepSeek-V3.2"
DATASET_PATH="/workspace/ms-swift/DSv32MTP_FT/sft_10.jsonl"
OUTPUT_DIR="/data/finetuned_layer60_hf"

# Training hyperparams
BATCH_SIZE=1
GRAD_ACCUM=8
EPOCHS=3
LR=2e-5
MAX_LENGTH=2048

# Trainable parameters: layer 60 + lm_head + model.norm
# 注意: DeepSeek-V3.2 是 MoE 模型，layer 60 包含 256 个 experts
# 只训练最后一层的 attention 和 shared_expert 可以减少参数量
TRAINABLE_REGEX="model\.layers\.60\.(self_attn|input_layernorm|post_attention_layernorm|mlp\.gate|mlp\.shared_experts)|lm_head|model\.norm"

# ===== Setup =====
mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "DeepSeek-V3.2 MTP Fine-tuning"
echo "============================================"
echo "Model: ${MODEL_DIR}"
echo "Dataset: ${DATASET_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Trainable regex: ${TRAINABLE_REGEX}"
echo "============================================"

# ===== Check dataset =====
if [[ ! -f "${DATASET_PATH}" ]]; then
    echo "[ERROR] Dataset not found: ${DATASET_PATH}"
    exit 1
fi

DATASET_LINES=$(wc -l < "${DATASET_PATH}")
echo "Dataset samples: ${DATASET_LINES}"

# ===== Train using swift sft =====
# 使用 HuggingFace 后端训练，适合快速验证
# 对于大规模训练，建议使用 megatron sft

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model "${MODEL_DIR}" \
    --model_type deepseek_v3_2 \
    --dataset "${DATASET_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --train_type full \
    --freeze_parameters_ratio 1.0 \
    --trainable_parameters_regex "${TRAINABLE_REGEX}" \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --num_train_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --max_length ${MAX_LENGTH} \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --bf16 true \
    --gradient_checkpointing true \
    --dataloader_num_workers 4 \
    --report_to none

echo ""
echo "============================================"
echo "Training completed!"
echo "Output saved to: ${OUTPUT_DIR}"
echo "============================================"
