#!/usr/bin/env bash
# ============================================================================
# DeepSeek-V3.2 MTP Last Layer Fine-tuning Script (Megatron Backend)
# 
# 目标: 微调最后一层 (layer 60) 相关参数以提高投机解码接受率
# 使用: ms-swift 的 megatron sft 命令
# 
# DeepSeek-V3.2 模型结构:
#   - num_hidden_layers: 61 (0-60)
#   - num_nextn_predict_layers: 1 (MTP uses layer 60)
#   - n_routed_experts: 256
#   - 总参数量: ~685B (其中MoE占大部分)
#
# 训练策略:
#   仅训练 layer 60 的 attention + shared_expert + gate
#   以及 lm_head 和 model.norm
# ============================================================================
set -euo pipefail

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

# ===== User-configurable =====
MODEL_DIR="/data/models/DeepSeek-V3.2"
DATASET_PATH="/workspace/ms-swift/DSv32MTP_FT/sft_10.jsonl"
OUTPUT_DIR="/data/finetuned_layer60"

# Parallelism (8x H200)
# TP=1, EP=8: 每个GPU加载 256/8=32 个experts
NPROC_PER_NODE=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TP=1      # Tensor Parallel
PP=1      # Pipeline Parallel
CP=1      # Context Parallel
EP=8      # Expert Parallel

# Training hyperparams
MICRO_BATCH=1
GLOBAL_BATCH=8
EPOCHS=3
LR=2e-5
MIN_LR=2e-6
MAX_LENGTH=2048
SAVE_INTERVAL=100

# Trainable parameters regex:
# - model.layers.60.self_attn.*: 最后一层的attention
# - model.layers.60.input_layernorm: 输入layernorm
# - model.layers.60.post_attention_layernorm: 后attention layernorm
# - model.layers.60.mlp.gate: MoE gate (决定expert选择)
# - model.layers.60.mlp.shared_experts: 共享expert
# - lm_head: 语言模型头
# - model.norm: 最终layernorm
#
# 注意: 不训练 model.layers.60.mlp.experts (256个experts太大)
TRAINABLE_REGEX="model\.layers\.60\.(self_attn|input_layernorm|post_attention_layernorm|mlp\.gate|mlp\.shared_experts)|lm_head|model\.norm"

# ===== Setup =====
mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "DeepSeek-V3.2 MTP Fine-tuning (Megatron)"
echo "============================================"
echo "Model: ${MODEL_DIR}"
echo "Dataset: ${DATASET_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Parallelism: TP=${TP}, PP=${PP}, CP=${CP}, EP=${EP}"
echo "Trainable: ${TRAINABLE_REGEX}"
echo "============================================"

# ===== Check dataset =====
if [[ ! -f "${DATASET_PATH}" ]]; then
    echo "[ERROR] Dataset not found: ${DATASET_PATH}"
    exit 1
fi

DATASET_LINES=$(wc -l < "${DATASET_PATH}")
echo "Dataset samples: ${DATASET_LINES}"
echo ""

# ===== Train using megatron sft =====
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
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 0 \
    --micro_batch_size ${MICRO_BATCH} \
    --global_batch_size ${GLOBAL_BATCH} \
    --recompute_granularity selective \
    --max_epochs ${EPOCHS} \
    --lr ${LR} \
    --lr_warmup_fraction 0.1 \
    --min_lr ${MIN_LR} \
    --save "${OUTPUT_DIR}" \
    --save_interval ${SAVE_INTERVAL} \
    --max_length ${MAX_LENGTH} \
    --num_workers 4 \
    --dataset_num_proc 4 \
    --no_save_optim true \
    --no_save_rng true \
    --packing true \
    --packing_max_seq_len ${MAX_LENGTH}

echo ""
echo "============================================"
echo "Training completed!"
echo "Output saved to: ${OUTPUT_DIR}"
echo ""
echo "Next step: Convert to HF format using 30_export_hf.sh"
echo "============================================"
