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

# pyre-unsafe

from typing import Dict, NamedTuple, Optional, Tuple

import torch


class SequentialFeatures(NamedTuple):
    # (B,) x int64. Requires past_lengths[i] > 0 \forall i.
    past_lengths: torch.Tensor
    # (B, N,) x int64. 0 denotes valid ids.
    past_ids: torch.Tensor
    # (B, N, D) x float.
    past_embeddings: Optional[torch.Tensor]
    # Implementation-specific payloads.
    # e.g., past timestamps, past event_types (e.g., clicks, likes), etc.
    past_payloads: Dict[str, torch.Tensor]


def movielens_seq_features_from_row(
    row: Dict[str, torch.Tensor],
    device: int,
    max_output_length: int,
) -> Tuple[SequentialFeatures, torch.Tensor, torch.Tensor]:
    """
    Args:
        row (Dict[str, torch.Tensor]):
        device (int): 默认情况来自rank, 本地环境中测试为0
        max_output_length (int): gr_output_length + 1为11

    Returns:
        Tuple[SequentialFeatures, torch.Tensor, torch.Tensor]: _description_
    """

    # 使用ml-1m数据集的情况下, row中变量的维度为
    #   historical_lengths: ([128]) 
    #   historical_ids: ([128, 200])
    #   historical_ratings: ([128, 200])
    #   historical_timestamps: ([128, 200])
    #   target_ids: ([128]) -> ([128, 1])
    #   target_ratings: ([128]) -> ([128, 1])
    #   target_timestamps: ([128]) -> ([128, 1])
    historical_lengths = row["history_lengths"].to(device)  # [B]
    historical_ids = row["historical_ids"].to(device)  # [B, N]
    historical_ratings = row["historical_ratings"].to(device)
    historical_timestamps = row["historical_timestamps"].to(device)
    target_ids = row["target_ids"].to(device).unsqueeze(1)  # [B, 1]
    target_ratings = row["target_ratings"].to(device).unsqueeze(1)
    target_timestamps = row["target_timestamps"].to(device).unsqueeze(1)
    if max_output_length > 0:
        B = historical_lengths.size(0)
        # ([128, 200] + [128, 11]) -> ([128, 211])
        historical_ids = torch.cat(
            [
                historical_ids,
                torch.zeros(
                    (B, max_output_length), dtype=historical_ids.dtype, device=device
                ),
            ],
            dim=1,
        )
        historical_ratings = torch.cat(
            [
                historical_ratings,
                torch.zeros(
                    (B, max_output_length),
                    dtype=historical_ratings.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        historical_timestamps = torch.cat(
            [
                historical_timestamps,
                torch.zeros(
                    (B, max_output_length),
                    dtype=historical_timestamps.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )

        # 对于批次中的每一个序列(每一行 i), 这行代码会将 target_timestamps[i, 0] 的值(即第i个序列的目标时间戳)
        # 写入到 historical_timestamps[i, historical_lengths[i]]的位置
        #
        # 换句话说，它将每个序列的目标时间戳放置在该序列原始历史记录的正后方. 如果historical_timestamps之前已经
        # 被填充了0(如代码中所示, 为了达到max_output_length), 那么这个目标时间戳就会覆盖掉那个位置的0.
        # 这是一种在序列数据中将目标信息整合到历史序列末尾的常见做法, 常用于为序列模型(如Transformer)准备输入,
        # 模型需要基于历史和当前目标的部分信息来预测未来的交互
        historical_timestamps.scatter_(
            dim=1,
            index=historical_lengths.view(-1, 1),
            src=target_timestamps.view(-1, 1),
        )
        # print(f"historical_ids.size()={historical_ids.size()}, historical_timestamps.size()={historical_timestamps.size()}")
    features = SequentialFeatures(
        past_lengths=historical_lengths,
        past_ids=historical_ids,
        past_embeddings=None,
        past_payloads={
            "timestamps": historical_timestamps,
            "ratings": historical_ratings,
        },
    )
    return features, target_ids, target_ratings
