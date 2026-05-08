#!/usr/bin/env bash
set -xeuo pipefail
export HYDRA_FULL_ERROR=1

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
MODEL_PATH="/vepfs/group03/user/ljm/work/taskllm_sft/output/checkpoint-500"
TRAIN_FILES="${VERL_DATA_DIR}/train.parquet"
VAL_FILES="${VERL_DATA_DIR}/test.parquet"

adv_estimator=grpo
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"
use_kl_in_reward=False
use_kl_loss=False
kl_loss_coef=0.0

enable_filter_groups=True
filter_groups_metric=seq_reward
max_num_gen_batches=10

max_prompt_length=256
max_response_length=128
n_resp_per_prompt=8
train_prompt_mini_bsz=512

lr=1e-6
log_every_n_steps=100

# Calculate save_freq: ~1 checkpoint per epoch
# total_samples / mini_bsz = steps_per_epoch
save_freq=3377

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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=128 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_model_len=${max_response_length} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=128 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    ++actor_rollout_ref.rollout.mode=async \
    custom_reward_function.path=verl/utils/reward_score/taskllm_exact_match.py \
    custom_reward_function.name=compute_score \
    reward_model.reward_manager=dapo \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='taskllm_dialogue_exact_match' \
    trainer.experiment_name='taskllm_grpo_exact_match' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=-1 \
    trainer.total_epochs=3 \
    trainer.val_before_train=False \
    +trainer.log_every_n_steps=${log_every_n_steps}