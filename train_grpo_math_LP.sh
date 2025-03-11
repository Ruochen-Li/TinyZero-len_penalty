export N_GPUS=8
export BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export DATA_DIR=data/math
export ROLLOUT_TP_SIZE=4
export EXPERIMENT_NAME=math-grpo-deepseek-7b-8h100-dlp
export VLLM_ATTENTION_BACKEND=XFORMERS

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/math_grpo_train_tiny_zero_8h100_LP.sh

