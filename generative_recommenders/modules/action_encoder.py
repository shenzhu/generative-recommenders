# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# pyre-strict

from typing import Dict, List, Optional, Tuple

import torch

from generative_recommenders.common import HammerModule
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged


class ActionEncoder(HammerModule):
    """
    Convert user actions and optional watchtime to fixed-length embeddings.

    Setup:
        - action_embedding_dim: 64
        - action_feature_name: "actions"
        - action_weights: [1, 2] (1 stands for "click" while 2 for "buy")
        - watchtime_feature_name: "watchtimes"
        - watchtime_to_action_thresholds_and_weights : [(10, 4), (60, 8)]
            - watchtime >= 10 seconds, considered as new action with weight 4
            - watchtime >= 60 seconds, considered as new action with weight 8
    
    After class initialization:
        - self._combined_action_weights: torch.tensor([1, 2, 4, 8])
        - self._num_action_types: 4
        - self._action_embedding_table: (4, 64) learnable matrix
        - self._target_action_embedding_table: (1, 4 * 64) learnable matrix
    """

    def __init__(
        self,
        action_embedding_dim: int,
        action_feature_name: str,
        action_weights: List[int],
        watchtime_feature_name: str = "",
        watchtime_to_action_thresholds_and_weights: Optional[
            List[Tuple[int, int]]
        ] = None,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._watchtime_feature_name: str = watchtime_feature_name
        self._action_feature_name: str = action_feature_name
        self._watchtime_to_action_thresholds_and_weights: List[Tuple[int, int]] = (
            watchtime_to_action_thresholds_and_weights
            if watchtime_to_action_thresholds_and_weights is not None
            else []
        )
        self.register_buffer(
            "_combined_action_weights",
            torch.tensor(
                action_weights
                + [x[1] for x in self._watchtime_to_action_thresholds_and_weights]
            ),
        )
        self._num_action_types: int = len(action_weights) + len(
            self._watchtime_to_action_thresholds_and_weights
        )
        self._action_embedding_dim = action_embedding_dim
        self._action_embedding_table: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty((self._num_action_types, action_embedding_dim)).normal_(
                mean=0, std=0.1
            ),
        )
        self._target_action_embedding_table: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty((1, self._num_action_types * action_embedding_dim)).normal_(
                mean=0, std=0.1
            ),
        )

    @property
    def output_embedding_dim(self) -> int:
        return self._action_embedding_dim * self._num_action_types

    def forward(
        self,
        max_uih_len: int,
        max_targets: int,
        uih_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        seq_embeddings: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Assume seq_payloads has the following content
        {
            "actions": - torch.tensor([1, 0, 2]),
            "watchtimes": torch.tensor([5, 15, 70])
        }

        Forward function will
        1. Extract actions and watchtimes from seq_payloads
            - seq_actions = torch.tensor([1, 0, 2])
            - seq_watchtimes = torch.tensor([5, 15, 70])
        2. Update seq_actions based on watch times
            For first threshold (10, 4)
            - watchtimes >= 10 -> torch.tensor([False, True, True])
            - convert to int64 and times weight -> torch.tensor([0, 4, 4])
            - bitwise_or -> torch.tensor([1, 4, 6])
            For second threshold (60, 8)
            - watchtimes >= 60 -> torch.tensor([False, False, True])
            - convert to int64 and times weight -> torch.tensor([0, 0, 8])
            - bitwise_or -> torch.tensor([1, 4, 14])
        3. 计算exploded_actions
            - seq_actions.unsqueeze(-1) : torch.tensor([[1], [4], [14]])
            - self._combined_action_weights.unsqueeze(0) : torch.tensor([[1, 2, 4, 8]])
            - torch.bitwise_and(seq_actions.unsqueeze(-1), self._combined_action_weights.unsqueeze(0)) :
                [[1 & 1, 1 & 2, 1 & 4, 1 & 8],   -> [[1, 0, 0, 0],
                [4 & 1, 4 & 2, 4 & 4, 4 & 8],   ->  [0, 0, 4, 0],
                [14 & 1, 14 & 2, 14 & 4, 14 & 8]]  ->  [0, 2, 4, 8]]
            - >0 转换为bool类型
                [[True, False, False, False],
                [False, False, True, False],
                [False, True, True, True]]
            - 这就是 exploded_actions. 它是一个(3, 4)的张量, 表示每个时间步(3个)激活了哪些行为类型(4种)
                - 行为1 (点击): 激活了第1种行为类型 (原始权重1)
                - 行为2 (观看>=10s): 激活了第3种行为类型 (观看时长权重4)
                - 行为3 (购买 + 观看>=10s + 观看>=60s): 激活了第2, 3, 4种行为类型 (原始权重2, 观看时长权重4, 观看时长权重8)
        """
        seq_actions = seq_payloads[self._action_feature_name]
        if len(self._watchtime_to_action_thresholds_and_weights) > 0:
            watchtimes = seq_payloads[self._watchtime_feature_name]
            for threshold, weight in self._watchtime_to_action_thresholds_and_weights:
                seq_actions = torch.bitwise_or(
                    seq_actions, (watchtimes >= threshold).to(torch.int64) * weight
                )
        exploded_actions = (
            torch.bitwise_and(
                seq_actions.unsqueeze(-1), self._combined_action_weights.unsqueeze(0)
            )
            > 0
        )
        action_embeddings = (
            exploded_actions.unsqueeze(-1) * self._action_embedding_table.unsqueeze(0)
        ).view(-1, self._num_action_types * self._action_embedding_dim)
        total_targets: int = seq_embeddings.size(0) - action_embeddings.size(0)
        action_embeddings = concat_2D_jagged(
            values_left=action_embeddings,
            values_right=self._target_action_embedding_table.tile(
                total_targets,
                1,
            ),
            max_len_left=max_uih_len,
            max_len_right=max_targets,
            offsets_left=uih_offsets,
            offsets_right=target_offsets,
            kernel=self.hammer_kernel(),
        )
        return action_embeddings
