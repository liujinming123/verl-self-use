#!/usr/bin/env bash
set -xeuo pipefail

# =============================================================================
# Multi-node Ray Training Script for Qwen 0.8B
# Configuration: 2 nodes, 1 GPU per node
# Environment Variables: MLP_WORKER_0_HOST, MLP_WORKER_0_PORT, MLP_ROLE_INDEX, 
#                        MLP_WORKER_NUM, MLP_WORKER_GPU
# =============================================================================

export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVTE_DISABLE_FP8=1
export HYDRA_FULL_ERROR=1
# Read environment variables
HEAD_NODE_IP="${MLP_WORKER_0_HOST:-localhost}"
RAY_PORT="${MLP_WORKER_0_PORT:-6379}"
NODE_RANK="${MLP_ROLE_INDEX:-0}"
NNODES="${MLP_WORKER_NUM:-1}"
GPUS_PER_NODE="${MLP_WORKER_GPU:-4}"

echo "=========================================="
echo "Node Configuration:"
echo "  HEAD_NODE_IP: $HEAD_NODE_IP"
echo "  RAY_PORT: $RAY_PORT"
echo "  NODE_RANK: $NODE_RANK"
echo "  NNODES: $NNODES"
echo "  GPUS_PER_NODE: $GPUS_PER_NODE"
echo "=========================================="

# Start Ray cluster
if [ "$NODE_RANK" == "0" ]; then
    echo "Starting Ray HEAD node..."
    ray start --head \
        --node-ip-address="$HEAD_NODE_IP" \
        --port="$RAY_PORT" \
        --num-gpus="$GPUS_PER_NODE" \
        --disable-usage-stats
    
    # Wait for all worker nodes to join
    echo "Waiting for $NNODES nodes to join the cluster..."
    while [ $(ray status 2>/dev/null | grep -c "node_") -lt "$NNODES" ]; do
        echo "  Current nodes: $(ray status 2>/dev/null | grep -c 'node_') / $NNODES"
        sleep 5
    done
    echo "All $NNODES nodes have joined the cluster!"
    ray status
else
    echo "Starting Ray WORKER node..."
    # Wait for head node to be ready
    sleep 10
    ray start --address="$HEAD_NODE_IP:$RAY_PORT" \
        --num-gpus="$GPUS_PER_NODE" \
        --disable-usage-stats
    echo "Worker node joined the cluster!"
    
    # Worker nodes wait for training to complete
    echo "Worker node waiting for training to finish..."
    while true; do
        if ! ray status &>/dev/null; then
            echo "Ray cluster disconnected, exiting..."
            break
        fi
        sleep 10
    done
    exit 0
fi

# ==================== Data Conversion ====================
# Convert SFT format data to verl RL format (run once)
DATA_DIR="/vepfs/group03/user/ljm/work/taskllm_sft/data"
VERL_DATA_DIR="${DATA_DIR}/verl_format"

if [ ! -f "${VERL_DATA_DIR}/train.parquet" ] || [ ! -f "${VERL_DATA_DIR}/test.parquet" ]; then
    echo "Converting data to verl format..."
    python3 recipe/dapo/convert_taskllm_sft_to_verl.py \
        --input_file "${DATA_DIR}/train.parquet" \
        --output_file "${VERL_DATA_DIR}/train.parquet" \
        --split train
    python3 recipe/dapo/convert_taskllm_sft_to_verl.py \
        --input_file "${DATA_DIR}/test.parquet" \
        --output_file "${VERL_DATA_DIR}/test.parquet" \
        --split test
fi

# ==================== Training Config ====================
MODEL_PATH="/vepfs/group03/user/ljm/base_model/Qwen/Qwen3-4B-Thinking-2507"

# Wait for model checkpoint directory to be available
while [ ! -d "${MODEL_PATH}" ]; do
    echo "Model directory not found: ${MODEL_PATH}, waiting 30 minutes..."
    sleep 1800
done
echo "Model directory found: ${MODEL_PATH}"

TRAIN_FILES="${VERL_DATA_DIR}/train.parquet"
VAL_FILES="${VERL_DATA_DIR}/test.parquet"

adv_estimator=grpo
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"
use_kl_in_reward=False
use_kl_loss=True
kl_loss_coef=0.001

enable_filter_groups=True
filter_groups_metric=seq_reward
max_num_gen_batches=10

max_prompt_length=64000
max_response_length=4096
n_resp_per_prompt=8
train_prompt_mini_bsz=64

lr=1e-6
log_every_n_steps=10

# Calculate save_freq: ~1 checkpoint per epoch
# total_samples / mini_bsz = steps_per_epoch
save_freq=500

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    +algorithm.filter_groups.enable=${enable_filter_groups} \
    +algorithm.filter_groups.metric=${filter_groups_metric} \
    +algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${train_prompt_mini_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    data.truncation='error' \
    +data.system_prompt_file="/vepfs/group03/user/ljm/work/taskllm_prompt/action_prompt.md" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
    actor_rollout_ref.actor.checkpoint.async_save=False \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length+max_response_length)) \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    ++actor_rollout_ref.rollout.mode=async \
    custom_reward_function.path=verl/utils/reward_score/taskllm_tag_param_match_v2.py \
    custom_reward_function.name=compute_score \
    reward_model.reward_manager=dapo \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='taskllm' \
    trainer.experiment_name='taskllm_grpo_tag_param_matchv2_sft_full_klloss_v2_think' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=100 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    +trainer.log_every_n_steps=${log_every_n_steps}