#!/usr/bin/env bash
set -xeuo pipefail
export HYDRA_FULL_ERROR=1

MODEL_PATH="/vepfs/group03/model-public/Qwen3-30B-A3B"
TRAIN_FILES="/vepfs/group03/user/ljm/work/qwen_grpo/data/verl_format/train_sft.parquet"
VAL_FILES="/vepfs/group03/user/ljm/work/qwen_grpo/data/verl_format/test_sft.parquet"

lr=1e-5
log_every_n_steps=10
train_batch_size=8

torchrun --standalone --nnodes=1 --nproc-per-node=8 \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${train_batch_size} \
    data.max_length=2048 \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.shuffle=True \
    data.prompt_key=prompt \
    data.response_key=response \
    data.apply_chat_template_kwargs.enable_thinking=False \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=True \
    model.use_fused_kernels=True \
    model.enable_gradient_checkpointing=True \
    engine=fsdp \
    optim=fsdp \
    optim.lr=${lr} \
    optim.lr_warmup_steps_ratio=0.01 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.warmup_style=cosine \
    engine.ulysses_sequence_parallel_size=8 \
    engine.strategy=fsdp2 \
    engine.fsdp_size=8 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='qwen_sft_car_dialogue' \
    trainer.experiment_name='sft_qwen3_30b_a3b_v1' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=647 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    +trainer.log_every_n_steps=${log_every_n_steps}
