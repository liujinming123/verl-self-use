#!/usr/bin/env bash
set -xeuo pipefail

# =============================================================================
# Online DPO Training Script for Qwen (task-llm)
# Based on verl/recipe/spin (Online DPO / SPIN)
# Configuration: 1 node, 4 GPUs
# =============================================================================

export FLASHINFER_DISABLE_VERSION_CHECK=1
export NVTE_DISABLE_FP8=1
export HYDRA_FULL_ERROR=1

# ==================== Data Conversion ====================
# Convert SFT format data to verl RL format (run once)
DATA_DIR="/vepfs/group03/user/ljm/work/taskllm_sft/data"
VERL_DATA_DIR="${DATA_DIR}/verl_format"

echo "Converting data to verl format..."
python3 recipe/dapo/convert_taskllm_sft_to_verl.py \
    --input_file "${DATA_DIR}/train_low_confidence.parquet" \
    --output_file "${VERL_DATA_DIR}/train_low_confidence.parquet" \
    --split train
python3 recipe/dapo/convert_taskllm_sft_to_verl.py \
    --input_file "${DATA_DIR}/test.parquet" \
    --output_file "${VERL_DATA_DIR}/test.parquet" \
    --split test

# ==================== DPO Config ====================
# DPO beta (temperature for DPO loss, higher = more conservative)
dpo_beta=0.1

# Loss type: "sigmoid" (standard DPO) or "ipo"
dpo_loss_type=sigmoid

# Reference model update frequency (update ref model every N iterations)
ref_update_freq=-1

# No PPO advantage estimation — DPO mode
adv_estimator=null

# ==================== Preference Filtering Config ====================
# Filter out prompts where all responses have the same score (no preference signal)
enable_filter_groups=True
filter_groups_metric=seq_reward
max_num_gen_batches=10

# ==================== Generation Config ====================
max_prompt_length=1500
max_response_length=128
# DPO needs exactly 2 responses per prompt (chosen vs rejected pair)
n_resp_per_prompt=2
train_prompt_mini_bsz=256

# ==================== Training Config ====================
lr=1e-6
log_every_n_steps=100

# Calculate save_freq: ~1 checkpoint per epoch
# total_samples / mini_bsz = steps_per_epoch
save_freq=2251

MODEL_PATH="/vepfs/group03/user/ljm/git_program/task-llm/checkpoints/merged"

# Wait for model checkpoint directory to be available
while [ ! -d "${MODEL_PATH}" ]; do
    echo "Model directory not found: ${MODEL_PATH}, waiting 30 minutes..."
    sleep 1800
done
echo "Model directory found: ${MODEL_PATH}"

TRAIN_FILES="${VERL_DATA_DIR}/train_low_confidence.parquet"
VAL_FILES="${VERL_DATA_DIR}/test.parquet"

# ==================== Reward Config ====================
# Use taskllm exact match reward for preference labeling
custom_reward_path=verl/utils/reward_score/taskllm_exact_match.py
custom_reward_name=compute_score

python3 -m recipe.spin.main_spin \
    algorithm.adv_estimator=${adv_estimator} \
    +algorithm.dpo_beta=${dpo_beta} \
    +algorithm.dpo_loss_type=${dpo_loss_type} \
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
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.dpo_beta=${dpo_beta} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    custom_reward_function.path=${custom_reward_path} \
    custom_reward_function.name=${custom_reward_name} \
    reward_model.reward_manager=dapo \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='taskllm_dialogue_dpo' \
    trainer.experiment_name='taskllm_dpo_exact_match_low_confidence' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    trainer.ref_update_freq=${ref_update_freq} \
    +trainer.log_every_n_steps=${log_every_n_steps} \
    "$@"