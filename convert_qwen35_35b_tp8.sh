#!/usr/bin/env bash
set -xeuo pipefail

# 转换脚本：将HF格式Qwen3.5-35B-A3B转换为TP=8的Megatron分布式检查点
# 输入路径：/vepfs/public/model-public/Qwen3.5-35B-A3B
# 输出路径：/vepfs/group03/user/ljm/base_model/Qwen/dist_ckpt

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 配置参数
INPUT_HF_PATH="/vepfs/public/model-public/Qwen3.5-35B-A3B"
OUTPUT_DIST_CKPT_PATH="/vepfs/group03/user/ljm/base_model/Qwen/Qwen3.5-35B-A3B_tp8"
TP_SIZE=8
PP_SIZE=1
EP_SIZE=1

echo "开始转换Qwen3.5-35B-A3B为TP=${TP_SIZE}分布式检查点..."
echo "输入路径: ${INPUT_HF_PATH}"
echo "输出路径: ${OUTPUT_DIST_CKPT_PATH}"

# 创建输出目录
mkdir -p ${OUTPUT_DIST_CKPT_PATH}

# 运行转换（8卡分布式）
torchrun --nproc_per_node=8 scripts/converter_hf_to_mcore.py \
  --hf_model_path "${INPUT_HF_PATH}" \
  --output_path "${OUTPUT_DIST_CKPT_PATH}" \
  --pp_size ${PP_SIZE} \
  --ep_size ${EP_SIZE} \
  --use_cpu_initialization \
  --trust_remote_code \
  --test

echo "转换完成！分布式检查点已生成到: ${OUTPUT_DIST_CKPT_PATH}"
echo "训练时配置参数："
echo "actor_rollout_ref.actor.megatron.use_dist_checkpointing=True"
echo "actor_rollout_ref.actor.megatron.dist_checkpointing_path=${OUTPUT_DIST_CKPT_PATH}"
