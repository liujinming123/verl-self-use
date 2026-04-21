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
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${train_batch_size} \
    ++data.micro_batch_size_per_gpu=1 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=3000 \
    data.truncation=error \
    ++data.apply_chat_template_kwargs.enable_thinking=False \
    model.partial_pretrain="${MODEL_PATH}" \
    ++model.enable_gradient_checkpointing=True \
    ++model.strategy=fsdp \
    ++model.enable_amp=True \
    ++model.lm_head_dtype=float32 \
    optim.lr=${lr} \
    optim.lr_warmup_steps_ratio=0.01 \
    optim.weight_decay=0.1 \
    'optim.betas=[0.9,0.95]' \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.lr_scheduler=cosine \
    use_remove_padding=True \
    ++ulysses_sequence_parallel_size=1 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='qwen_sft_car_dialogue' \
    trainer.experiment_name='sft_qwen3_30b_a3b_v1' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=647 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    +trainer.log_every_n_steps=${log_every_n_steps}