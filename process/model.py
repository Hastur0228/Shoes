"""
Deep Graph Convolutional Neural Network (DGCNN) for point cloud processing.

This implementation is adapted for this project from the provided reference
`model.py` and exposes only the DGCNN model along with the required helpers.

Input shape: (batch_size, 3, num_points)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Compute k-nearest neighbors indices for each point in the batch.

    Args:
        x: Tensor of shape (batch_size, feature_dim, num_points)
        k: Number of neighbors

    Returns:
        Tensor of indices with shape (batch_size, num_points, k)
    """
    # x -> (B, C, N)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # -(||xi - xj||^2)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx


def _get_graph_feature(
    x: torch.Tensor,
    k: int = 20,
    idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Construct edge features for each point's kNN graph.

    For each point, returns concatenation of (x_j - x_i, x_i) over k neighbors.

    Args:
        x: Tensor of shape (batch_size, feature_dim, num_points)
        k: Number of neighbors
        idx: Optional precomputed kNN indices

    Returns:
        Edge features of shape (batch_size, 2*feature_dim, num_points, k)
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = _knn(x, k=k)  # (B, N, k)

    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base  # (B, N, k) + (B, 1, 1)
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (B, N, C)

    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # (x_j - x_i, x_i)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


@dataclass
class DGCNNConfig:
    k: int = 20
    emb_dims: int = 1024
    dropout: float = 0.5
    output_channels: int = 40
    input_dims: int = 3  # 输入点的通道数（默认 XYZ）


class DGCNN(nn.Module):
    """DGCNN 编码器 + 全连接解码头。

    默认作为分类/回归头。如果用于点云到点云生成，将结合下方的
    `PointSetGenerator` 作为解码器。
    """

    def __init__(self, config: DGCNNConfig):
        super().__init__()
        self.config = config
        self.k = config.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(config.emb_dims)

        # conv1 的输入通道为 2 * input_dims（边特征拼接 (x_j - x_i, x_i)）
        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * config.input_dims, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, config.emb_dims, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(0.2)
        )

        self.linear1 = nn.Linear(config.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=config.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=config.dropout)
        self.linear3 = nn.Linear(256, config.output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        x = _get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = _get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = _get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = _get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), dim=1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class PointSetGenerator(nn.Module):
    """将全局特征映射为目标点云 (3, N) 的生成头。

    结构：MLP 将嵌入向量映射为 (N*3)，再 reshape 成 (B, 3, N)。
    """

    def __init__(self, embedding_dim: int, num_points: int):
        super().__init__()
        hidden = max(512, embedding_dim // 2)
        self.num_points = num_points
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden),  # 使用 max+avg 拼接后的维度
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_points * 3),
        )

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        # 输入形状 (B, 2*embedding_dim)
        out = self.mlp(pooled_features)  # (B, N*3)
        out = out.view(out.size(0), self.num_points, 3).transpose(1, 2).contiguous()  # (B, 3, N)
        return out


class DGCNNPointCloud2PointCloud(nn.Module):
    """端到端：输入足部点云 (B, 3, N)，输出鞋垫点云 (B, 3, N)。"""

    def __init__(self, k: int = 20, emb_dims: int = 1024, num_points: int = 2048, dropout: float = 0.5, input_dims: int = 3):
        super().__init__()
        self.encoder, cfg = build_dgcnn_model(k=k, emb_dims=emb_dims, dropout=dropout, output_channels=40, input_dims=input_dims)
        # 替换分类头为 identity，保留到 pooled 特征
        # 我们重用 encoder 的 forward 到 pooling 前一步，因此在这里复制必要层
        self.encoder_linear1 = self.encoder.linear1
        self.encoder_bn6 = self.encoder.bn6
        self.encoder_dp1 = self.encoder.dp1
        self.encoder_linear2 = self.encoder.linear2
        self.encoder_bn7 = self.encoder.bn7
        self.encoder_dp2 = self.encoder.dp2
        # 生成器头
        self.generator = PointSetGenerator(embedding_dim=emb_dims, num_points=num_points)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 复制 DGCNN 的编码到池化拼接
        batch_size = x.size(0)
        k = self.encoder.k

        xg = _get_graph_feature(x, k=k)
        xg = self.encoder.conv1(xg)
        x1 = xg.max(dim=-1, keepdim=False)[0]

        xg = _get_graph_feature(x1, k=k)
        xg = self.encoder.conv2(xg)
        x2 = xg.max(dim=-1, keepdim=False)[0]

        xg = _get_graph_feature(x2, k=k)
        xg = self.encoder.conv3(xg)
        x3 = xg.max(dim=-1, keepdim=False)[0]

        xg = _get_graph_feature(x3, k=k)
        xg = self.encoder.conv4(xg)
        x4 = xg.max(dim=-1, keepdim=False)[0]

        xg = torch.cat((x1, x2, x3, x4), dim=1)
        xg = self.encoder.conv5(xg)

        x1p = F.adaptive_max_pool1d(xg, 1).view(batch_size, -1)
        x2p = F.adaptive_avg_pool1d(xg, 1).view(batch_size, -1)
        pooled = torch.cat((x1p, x2p), dim=1)  # (B, 2*emb_dims)

        # 通过生成器生成目标点云
        pred = self.generator(pooled)  # (B, 3, N)
        return pred


def build_dgcnn_model(
    k: int = 20,
    emb_dims: int = 1024,
    dropout: float = 0.5,
    output_channels: int = 40,
    input_dims: int = 3,
) -> Tuple[DGCNN, DGCNNConfig]:
    """Factory to construct a DGCNN model and its config.

    Returns:
        (model, config)
    """
    config = DGCNNConfig(k=k, emb_dims=emb_dims, dropout=dropout, output_channels=output_channels, input_dims=input_dims)
    return DGCNN(config), config


__all__ = [
    "DGCNN",
    "DGCNNConfig",
    "build_dgcnn_model",
    "PointSetGenerator",
    "DGCNNPointCloud2PointCloud",
]


