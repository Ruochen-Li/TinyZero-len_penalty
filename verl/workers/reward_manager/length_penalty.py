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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from collections import defaultdict

#from lighteval.metrics.dynamic_metrics import (
#    ExprExtractionConfig,
#    LatexExtractionConfig,
#    multilingual_extractive_match_metric,
#)
#from lighteval.tasks.requests import Doc
#from lighteval.utils.language import Language


class LPRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source') -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

        self.max_length = 3072
        self.preset_acc = 0.74

#        latex_gold_metric = multilingual_extractive_match_metric(
#            language=Language.ENGLISH,
#            fallback_mode="first_match",
#            precision=5,
#            gold_extraction_target=(LatexExtractionConfig(),),
#            # Match boxed first before trying other regexes
#            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
#            aggregation_function=max,
#        )
#        self.compute_score = compute_score or latex_gold_metric

    def __call__(self, data: DataProto, return_dict=False, validation_acc: float=-1):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

#            doc = Doc(
#                task_name='lp',
#                query=prompt_ids,
#                choices=[ground_truth],
#                gold_index=0,
#            )
#            score = self.compute_score.compute(
#                golds=[ground_truth],
#                predictions=[response_str],
#                formatted_doc=doc
#            )
#            score = list(score.values())[0]

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            # apply length_penalty only during training
            if -1 != val_acc:
                acc_ratio = min(1, val_acc / self.preset_acc) # [0,1]
                alpha = 0.9 + 0.1 * (1 - acc_ratio)
                length_ratio = min(1, valid_response_length / self.max_length) # [0,1]
                length_reward = 1 - min(acc_ratio ** 128, length_ratio)
                reward = reward * alpha + 1e-7 * length_reward

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("======================================================")
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
