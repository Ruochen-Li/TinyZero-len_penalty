set -x

python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=256 \
data.max_prompt_length=1024 \
data.max_response_length=3072 \
data.filter_overlong_prompts=True \
data.truncation='error' \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.ppo_mini_batch_size=128 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.001 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.fsdp_config.param_offload=False \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=80 \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
actor_rollout_ref.rollout.n=4 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=80 \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
algorithm.kl_ctrl.kl_coef=0.001 \
trainer.critic_warmup=0 \
trainer.logger=['console'] \
trainer.project_name='TinyZero' \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.n_gpus_per_node=8 \
trainer.nnodes=1 \
trainer.save_freq=30 \
trainer.test_freq=30 \
trainer.total_epochs=15 2>&1 | tee verl_grpo_math_7b_dlp.log

