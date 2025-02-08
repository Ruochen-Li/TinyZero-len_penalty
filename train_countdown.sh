export N_GPUS=2
export BASE_MODEL=Qwen/Qwen2.5-3B
export DATA_DIR=data/countdown
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b-2a100
export VLLM_ATTENTION_BACKEND=XFORMERS

CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/countdown_train_tiny_zero_2a100.sh

