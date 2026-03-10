#!/usr/bin/env bash
set -xeuo pipefail
export HYDRA_FULL_ERROR=1

MODEL_PATH="/vepfs/public/model-public/Qwen3-0.6B"
TRAIN_FILES="/vepfs/group03/user/ljm/work/qwen_grpo/data/verl_format/train_sft.parquet"
VAL_FILES="/vepfs/group03/user/ljm/work/qwen_grpo/data/verl_format/test_sft.parquet"

torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=4 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=2048 \
    data.truncation=error \
    ++data.apply_chat_template_kwargs.enable_thinking=False \
    model.partial_pretrain="${MODEL_PATH}" \
    ++model.enable_amp=True \
    ++model.lm_head_dtype=float32 \
    optim.lr=2e-5 \
    optim.lr_warmup_steps_ratio=0.05 \
    optim.weight_decay=0.01 \
    'optim.betas=[0.9,0.95]' \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.lr_scheduler=cosine \
    trainer.logger='["console"]' \
    trainer.project_name='qwen_sft_car_dialogue' \
    trainer.experiment_name='sft_qwen3_0.6b_test' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=3 \
    trainer.save_freq=10 \
    trainer.test_freq=10
