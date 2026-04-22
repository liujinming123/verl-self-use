#!/usr/bin/env bash
set -xeuo pipefail
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

# Read environment variables
HEAD_NODE_IP="${MLP_WORKER_0_HOST:-localhost}"
RAY_PORT="${MLP_WORKER_0_PORT:-6379}"
NODE_RANK="${MLP_ROLE_INDEX:-0}"
NNODES="${MLP_WORKER_NUM:-1}"
GPUS_PER_NODE="${MLP_WORKER_GPU:-8}"

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

# Only head node executes the training
MODEL_PATH="/vepfs/public/model-public/Qwen3.5-35B-A3B"
TRAIN_FILES="/vepfs/group03/user/ljm/work/qwen_grpo/data/verl_format/train_new_v2.parquet"
VAL_FILES="/vepfs/group03/user/ljm/work/qwen_grpo/data/verl_format/test_new_v2.parquet"

TP=${TP:-2}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-4}
ETP=${ETP:-1}
GEN_TP=${GEN_TP:-8}

ALL_OFFLOAD=${ALL_OFFLOAD:-True}

adv_estimator=grpo
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"
use_kl_in_reward=False
use_kl_loss=False
kl_loss_coef=0.0

enable_filter_groups=True
filter_groups_metric=seq_reward
max_num_gen_batches=16

max_prompt_length=5000
max_response_length=15000
n_resp_per_prompt=16
train_prompt_mini_bsz=8
train_bsz=8

lr=1e-6
log_every_n_steps=10

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    +algorithm.filter_groups.enable=${enable_filter_groups} \
    +algorithm.filter_groups.metric=${filter_groups_metric} \
    +algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${train_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.dataloader_num_workers=8 \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1024 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
    actor_rollout_ref.actor.checkpoint.async_save=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
    actor_rollout_ref.actor.megatron.use_remove_padding=False \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP} \
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD} \
    actor_rollout_ref.actor.megatron.dtype=bfloat16 \
    actor_rollout_ref.actor.megatron.sequence_parallel=True \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.max_model_len=${max_response_length} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=1024 \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.rollout.over_sample_rate=0 \
    ++actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.hybrid_engine=True \
    custom_reward_function.path=verl/utils/reward_score/qwen_grpo_car_dialogue.py \
    custom_reward_function.name=compute_score \
    reward_model.reward_manager=dapo \
    +reward_model.num_examine=1 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='qwen_grpo_car_dialogue_dapo' \
    trainer.experiment_name='prod_qwen35_35b_a3b_rl_v3' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1287 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    +trainer.log_every_n_steps=${log_every_n_steps}
