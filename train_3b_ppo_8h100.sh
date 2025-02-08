#!/bin/bash
# alias python='/home/weiji/anaconda3/envs/zero/bin/python'
# alias python3='/home/weiji/anaconda3/envs/zero/bin/python3'
# alias pip='/home/weiji/anaconda3/envs/zero/bin/pip'

export N_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ray stop --force && ray start --head
export BASE_MODEL="model/Qwen2.5-3B"
export DATA_DIR="data/countdown"
export ROLLOUT_TP_SIZE=8
export EXPERIMENT_NAME=countdown-qwen2.5-3b-lp
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero_8h100_ppo.sh