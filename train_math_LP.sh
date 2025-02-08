export N_GPUS=4
export BASE_MODEL=Qwen/Qwen2.5-3B
export DATA_DIR=data/math
export ROLLOUT_TP_SIZE=4
export EXPERIMENT_NAME=math-qwen2.5-3b-4a100-lp
export VLLM_ATTENTION_BACKEND=XFORMERS

CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/math_train_tiny_zero_4a100_LP.sh

