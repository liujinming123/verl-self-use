#!/usr/bin/env bash
set -xeuo pipefail

# Small model for testing
MODEL_PATH="/vepfs/group03/model-public/Qwen3-0.6B"
TRAIN_FILES="/vepfs/group03/user/ljm/work/qwen_grpo/data/verl_format/train.parquet"
VAL_FILES="/vepfs/group03/user/ljm/work/qwen_grpo/data/verl_format/test.parquet"

adv_estimator=grpo
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"
use_kl_in_reward=False
use_kl_loss=False
kl_loss_coef=0.0

enable_filter_groups=True
filter_groups_metric=seq_reward
max_num_gen_batches=2  # Reduced for testing

max_prompt_length=1024  # Reduced for small model
max_response_length=1024  # Reduced for small model
n_resp_per_prompt=2  # Reduced for testing
train_prompt_mini_bsz=8  # Adjusted to be divisible by 8 workers

lr=1e-6
log_every_n_steps=5  # Increased logging frequency for testing

# Tool configuration
TOOL_CONFIG_PATH="/vepfs/group03/user/ljm/git_program/verl/recipe/dapo/tool_config.yaml"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${train_prompt_mini_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    data.truncation='left' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.strategy=fsdp \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    +algorithm.filter_groups.enable=${enable_filter_groups} \
    +algorithm.filter_groups.metric=${filter_groups_metric} \
    +algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    custom_reward_function.path=verl/utils/reward_score/qwen_grpo_car_dialogue.py \
    custom_reward_function.name=compute_score \
    reward_model.reward_manager=dapo \
    +reward_model.num_examine=1 \
    trainer.logger='["console"]' \
    trainer.project_name='qwen_grpo_car_dialogue_dapo_test' \
    trainer.experiment_name='test_qwen3_0.6b' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    +trainer.log_every_n_steps=${log_every_n_steps} \
    ++actor_rollout_ref.rollout.mode=async \
    ++actor_rollout_ref.rollout.multi_turn.enable=True \
    ++actor_rollout_ref.rollout.multi_turn.max_user_turns=2 \
    ++actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    ++actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    ++actor_rollout_ref.rollout.multi_turn.max_tool_response_length=500 \
    ++actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=left \
    ++actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}" \
    ++actor_rollout_ref.rollout.multi_turn.format=gpt-oss