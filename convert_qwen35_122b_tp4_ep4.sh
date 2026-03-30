#!/usr/bin/env bash
set -xeuo pipefail

# 转换脚本：将HF格式Qwen3.5-122B-A10B转换为适配16卡训练的Megatron分布式检查点
# 最优并行配置：TP=4（张量并行） + EP=4（专家并行） + PP=1（流水线并行），完全适配2机16卡A100场景
# 输入路径：/vepfs/public/model-public/Qwen3.5-122B-A10B
# 输出路径：/vepfs/group03/user/ljm/base_model/Qwen/Qwen3.5-122B-A10B_tp4_ep4

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 配置参数（和训练时的并行配置严格匹配）
INPUT_HF_PATH="/vepfs/public/model-public/Qwen3.5-122B-A10B"
OUTPUT_DIST_CKPT_PATH="/vepfs/group03/user/ljm/base_model/Qwen/Qwen3.5-122B-A10B_tp4_ep4"
PP_SIZE=1   # 流水线并行度，122B单PP足够
EP_SIZE=4   # 专家并行度，和训练时保持一致

echo "开始转换Qwen3.5-122B-A10B为TP=4 EP=4分布式检查点..."
echo "输入路径: ${INPUT_HF_PATH}"
echo "输出路径: ${OUTPUT_DIST_CKPT_PATH}"
echo "并行配置: PP=${PP_SIZE} EP=${EP_SIZE} 适配训练TP=4 EP=4 16卡训练场景"

# 创建输出目录
mkdir -p ${OUTPUT_DIST_CKPT_PATH}

# 运行转换（8卡分布式，建议在8卡A100 80G机器上运行，16卡更快）
torchrun --nproc_per_node=8 scripts/converter_hf_to_mcore.py \
  --hf_model_path "${INPUT_HF_PATH}" \
  --output_path "${OUTPUT_DIST_CKPT_PATH}" \
  --pp_size ${PP_SIZE} \
  --ep_size ${EP_SIZE} \
  --use_cpu_initialization \
  --trust_remote_code \
  --test

echo "转换完成！分布式检查点已生成到: ${OUTPUT_DIST_CKPT_PATH}"
echo "16卡训练时配置参数："
echo "actor_rollout_ref.actor.megatron.use_dist_checkpointing=True"
echo "actor_rollout_ref.actor.megatron.dist_checkpointing_path=${OUTPUT_DIST_CKPT_PATH}"
echo "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4"
echo "actor_rollout_ref.actor.megatron.expert_model_parallel_size=4"
echo "actor_rollout_ref.rollout.tensor_model_parallel_size=4  # 和训练TP匹配，零权重同步开销"
