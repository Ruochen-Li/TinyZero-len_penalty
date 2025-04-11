# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
import torch.nn as nn
from verl.utils.reward_score import gsm8k, math, multiply, countdown
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from verl.utils.tracking import Tracking
from omegaconf import OmegaConf

#def _select_rm_score_fn(data_source):
#    if data_source == 'openai/gsm8k':
#        return gsm8k.compute_score
#    elif data_source == 'DigitalLearningGmbH/MATH-lighteval':
#        return math.compute_score
#    elif "multiply" in data_source or "arithmetic" in data_source:
#        return multiply.compute_score
#    elif "countdown" in data_source:
#        return countdown.compute_score
#    else:
#        raise NotImplementedError

#class RewardManager():
#    """The reward manager.
#    """
#
#    def __init__(self, tokenizer, num_examine) -> None:
#        self.tokenizer = tokenizer
#        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
#        self.max_length = 1024
#        self.n_step = 0
#        self.acc_avg = 0
#        self.preset_acc = 0.92
##        self.config = config
##        self.logger = Tracking(project_name=self.config.trainer.project_name,
##                               experiment_name=self.config.trainer.experiment_name,
##                               default_backend=self.config.trainer.logger,
##                               config=OmegaConf.to_container(self.config, resolve=True))
##        self.name = name
#
#        # Learnable weights for length and reflection penalties
#        # self.w_length = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))  # Weight for length penalty
#        # self.w_reflection = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))  # Weight for reflection penalty
#
#    def __call__(self, data: DataProto):
#        """We will expand this function gradually based on the available datasets"""
#
#        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
#        if 'rm_scores' in data.batch.keys():
#            return data.batch['rm_scores']
#
#        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
#
#        already_print_data_sources = {}
#
#        for i in range(len(data)):
#            data_item = data[i]  # DataProtoItem
##            print(data_item)
#
#            prompt_ids = data_item.batch['prompts']
#
#            prompt_length = prompt_ids.shape[-1]
#
#            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
#            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
#
#            response_ids = data_item.batch['responses']
#            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
#            valid_response_ids = response_ids[:valid_response_length]
#
#            # decode
#            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
#            sequences_str = self.tokenizer.decode(sequences)
#
#            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
#
#            # select rm_score
#            data_source = data_item.non_tensor_batch['data_source']
#            compute_score_fn = _select_rm_score_fn(data_source)
#
#            # Extract number of reflections from metadata (if available)
#            # num_reflections = data_item.non_tensor_batch.get('num_reflections', 0)
#
#            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
#            self.n_step += 1 # or the batch size
#            self.acc_avg += 1 / self.n_step * (score - self.acc_avg)
#            acc_diff_ratio = (self.preset_acc - self.acc_avg) / self.preset_acc # [-infinity, 1]
#            alpha = 0.8 + 0.2 * max(0, acc_diff_ratio)
#
#            # Length penalty (learnable weight applied)
#            # length_penalty = 0.07 * (1 - valid_response_length / self.max_length)
#            length_penalty = (1 - valid_response_length / self.max_length)
#            beta = 0.1 * min(1, 1 - acc_diff_ratio)
#            
#            # Reflection penalty (learnable weight applied)
#            # reflection_penalty = self.w_reflection * num_reflections
#
#            # Adjust reward
#            # adjusted_score = score + length_penalty - reflection_penalty
#            # adjusted_score = score + length_penalty
#            adjusted_score = alpha * score + beta * length_penalty
#            # adjusted_score = score
##            print(alpha, score)
##            print(beta, length_penalty)
##            log_data = {
##                f'{self.name}/accuracy': self.acc_avg,
##                f'{self.name}/alpha': alpha,
##                f'{self.name}/beta': beta,
##                f'{self.name}/score': score,
##                f'{self.name}/length_penalty': length_penalty,
##                f'{self.name}/reward': adjusted_score,
##            }
##            self.logger.log(data=log_data, step=self.n_step)
#
#            reward_tensor[i, valid_response_length - 1] = adjusted_score
#
#            if data_source not in already_print_data_sources:
#                already_print_data_sources[data_source] = 0
#
#            if already_print_data_sources[data_source] < self.num_examine:
#                already_print_data_sources[data_source] += 1
#                print(sequences_str)
#
#        return reward_tensor

# class RewardManager():
#     """The reward manager.
#     """

#     def __init__(self, tokenizer, num_examine) -> None:
#         self.tokenizer = tokenizer
#         self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
#         self.max_length = 1024

#         # Learnable weights for length and reflection penalties
#         self.w_length = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))  # Weight for length penalty
#         self.w_reflection = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))  # Weight for reflection penalty

#     def __call__(self, data: DataProto):
#         """We will expand this function gradually based on the available datasets"""

#         # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
#         if 'rm_scores' in data.batch.keys():
#             return data.batch['rm_scores']

#         reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

#         already_print_data_sources = {}

#         for i in range(len(data)):
#             data_item = data[i]  # DataProtoItem

#             prompt_ids = data_item.batch['prompts']

#             prompt_length = prompt_ids.shape[-1]

#             valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
#             valid_prompt_ids = prompt_ids[-valid_prompt_length:]

#             response_ids = data_item.batch['responses']
#             valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
#             valid_response_ids = response_ids[:valid_response_length]

#             # decode
#             sequences = torch.cat((valid_prompt_ids, valid_response_ids))
#             sequences_str = self.tokenizer.decode(sequences)

#             ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

#             # select rm_score
#             data_source = data_item.non_tensor_batch['data_source']
#             compute_score_fn = _select_rm_score_fn(data_source)

#             score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)

#             # Length penalty (learnable weight applied)
#             length_penalty = self.w_length * (1 - valid_response_length / self.max_length)
            
#             # Reflection penalty (learnable weight applied)
#             reflection_penalty = self.w_reflection * num_reflections

#             # Adjust reward
#             adjusted_score = score + length_penalty - reflection_penalty

#             reward_tensor[i, valid_response_length - 1] = adjusted_score

#             if data_source not in already_print_data_sources:
#                 already_print_data_sources[data_source] = 0

#             if already_print_data_sources[data_source] < self.num_examine:
#                 already_print_data_sources[data_source] += 1
#                 print(sequences_str)

#         return reward_tensor


import ray
import hydra


def get_custom_reward_fn(config):
    import importlib.util, os

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config):
    from verl.utils.fs import copy_to_local
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    import json
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer, hf_processor
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

#    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # Note that we always use function-based RM for validation
#    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == 'naive':
        from verl.workers.reward_manager import NaiveRewardManager
        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == 'prime':
        from verl.workers.reward_manager import PrimeRewardManager
        reward_manager_cls = PrimeRewardManager
    elif reward_manager_name == 'length_penalty':
        from verl.workers.reward_manager import LPRewardManager
        reward_manager_cls = LPRewardManager
    else:
        raise NotImplementedError

    compute_score = get_custom_reward_fn(config)
    reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)

    # Note that we always use function-based RM for validation
    val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            processor=processor,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
