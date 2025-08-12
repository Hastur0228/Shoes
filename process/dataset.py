from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


_FOOT_PATTERN = re.compile(r"^(\d+)_foot_([LR])\.npy$")


def _normalize_points(points: np.ndarray) -> np.ndarray:
    """规范化点云：
    - XYZ: 零均值并缩放至单位球
    - Normal(若存在): 单位化至长度 1

    其他通道（如颜色/强度）保持不变。
    """
    assert points.ndim == 2 and points.shape[1] >= 3, "points 需为 (N, C>=3)"
    xyz = points[:, :3]
    centroid = xyz.mean(axis=0, keepdims=True)
    xyz = xyz - centroid
    scale = np.max(np.linalg.norm(xyz, axis=1)) + 1e-9
    xyz = xyz / scale
    out = points.copy()
    out[:, :3] = xyz
    # 单位化法向（若存在）
    if out.shape[1] >= 6:
        normals = out[:, 3:6]
        norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
        out[:, 3:6] = normals / norms
    return out


def normalize_points_with_stats(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """对点云进行规范化并返回归一化后的点、以及还原所需的统计量。

    返回:
        normalized: 归一化后的点云 (N, C)
        centroid: 原始 XYZ 的质心 (1, 3)
        scale: 原始 XYZ 到单位球的缩放因子 (float)
    """
    assert points.ndim == 2 and points.shape[1] >= 3
    xyz = points[:, :3]
    centroid = xyz.mean(axis=0, keepdims=True)
    xyz = xyz - centroid
    scale = float(np.max(np.linalg.norm(xyz, axis=1)) + 1e-9)
    xyz = xyz / scale
    out = points.copy()
    out[:, :3] = xyz
    if out.shape[1] >= 6:
        normals = out[:, 3:6]
        norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
        out[:, 3:6] = normals / norms
    return out, centroid.astype(np.float32), scale


def denormalize_xyz(points_xyz: np.ndarray, centroid: np.ndarray, scale: float) -> np.ndarray:
    """将规范化到单位球的 XYZ 还原到原坐标系。"""
    return points_xyz * scale + centroid


def _resample_points(points: np.ndarray, num_points: int, rng: np.random.RandomState) -> np.ndarray:
    """Randomly sample or upsample to target num_points."""
    n = points.shape[0]
    if n == num_points:
        return points
    if n > num_points:
        idx = rng.choice(n, size=num_points, replace=False)
        return points[idx]
    # n < num_points: upsample by random repeat
    extra = rng.choice(n, size=num_points - n, replace=True)
    return np.concatenate([points, points[extra]], axis=0)


def _upsample_points_by_interpolation(points: np.ndarray, target_points: int, rng: np.random.RandomState) -> np.ndarray:
    """Upsample points to target size via linear interpolation between random pairs.

    - For XYZ: linear blend p = a * p_i + (1-a) * p_j
    - For normals (if present): blend then renormalize
    - Other channels (if any beyond 6) are linearly blended
    """
    n, c = points.shape
    if n >= target_points:
        return points[:target_points]
    need = target_points - n
    idx_a = rng.randint(0, n, size=need)
    idx_b = rng.randint(0, n, size=need)
    alpha = rng.rand(need, 1).astype(points.dtype)

    base_a = points[idx_a]
    base_b = points[idx_b]
    new_pts = alpha * base_a + (1.0 - alpha) * base_b

    # If normals exist, renormalize them
    if c >= 6:
        normals = new_pts[:, 3:6]
        norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
        new_pts[:, 3:6] = normals / norms

    return np.concatenate([points, new_pts], axis=0)


def resample_points_interpolate(points: np.ndarray, num_points: int, rng: np.random.RandomState) -> np.ndarray:
    """Resample to exactly num_points using downsampling or interpolation-based upsampling.

    - If shrinking: random subset without replacement.
    - If expanding: create new points by linear interpolation between random pairs, preserving normal unit length if present.
    """
    n = points.shape[0]
    if n == num_points:
        return points
    if n > num_points:
        idx = rng.choice(n, size=num_points, replace=False)
        return points[idx]
    return _upsample_points_by_interpolation(points, num_points, rng)


@dataclass
class FootInsoleDatasetConfig:
    root_dir: str = "data/pointcloud"
    feet_subdir: str = "feet"
    insoles_subdir: str = "insoles"
    num_points: int = 2048
    normalize: bool = True
    seed: int = 42
    use_normals: bool = False  # 若为 True，且数据通道>=6，则使用 XYZ+Normal 共6维
    side_filter: str | None = None  # 可为 'L' 或 'R'，None 表示两侧都用


class FootInsoleDataset(Dataset):
    """Dataset yielding (foot_points, insole_points) as tensors of shape (3, N)."""

    def __init__(self, config: FootInsoleDatasetConfig, split: Optional[Tuple[float, float]] = None, is_train: bool = True):
        super().__init__()
        self.config = config
        self.is_train = is_train
        self.rng = np.random.RandomState(config.seed if is_train else config.seed + 1)

        feet_dir = os.path.join(config.root_dir, config.feet_subdir)
        insoles_dir = os.path.join(config.root_dir, config.insoles_subdir)

        all_pairs: List[Tuple[str, str]] = []
        for fname in sorted(os.listdir(feet_dir)):
            m = _FOOT_PATTERN.match(fname)
            if not m:
                continue
            pid, side = m.group(1), m.group(2)
            if self.config.side_filter is not None and side != self.config.side_filter:
                continue
            foot_path = os.path.join(feet_dir, fname)
            insole_name = f"{pid}_insole_{side}.npy"
            insole_path = os.path.join(insoles_dir, insole_name)
            if os.path.isfile(insole_path):
                all_pairs.append((foot_path, insole_path))

        if not all_pairs:
            raise RuntimeError("未在数据集中找到匹配的足部与鞋垫点云对")

        # Optional split: (train_ratio, val_ratio). Default: use all.
        if split is not None:
            train_ratio, val_ratio = split
            assert abs(train_ratio + val_ratio - 1.0) < 1e-6, "split 应为 (train, val) 且和为 1"
            total = len(all_pairs)
            train_n = int(total * train_ratio)
            if is_train:
                self.pairs = all_pairs[:train_n]
            else:
                self.pairs = all_pairs[train_n:]
        else:
            self.pairs = all_pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        foot_path, insole_path = self.pairs[idx]
        foot = np.load(foot_path)  # (N, C)
        insole = np.load(insole_path)  # (M, C)

        # 选择通道：优先使用 6 维 (XYZ+Normal)，否则退化到 3 维 XYZ
        if self.config.use_normals and foot.shape[1] >= 6 and insole.shape[1] >= 6:
            foot = foot[:, :6]
            insole = insole[:, :6]
        else:
            foot = foot[:, :3]
            insole = insole[:, :3]

        if self.config.normalize:
            foot = _normalize_points(foot)
            insole = _normalize_points(insole)

        foot = _resample_points(foot, self.config.num_points, self.rng)
        insole = _resample_points(insole, self.config.num_points, self.rng)

        # To (C, N)
        foot_t = torch.from_numpy(foot.astype(np.float32)).transpose(0, 1)
        insole_t = torch.from_numpy(insole.astype(np.float32)).transpose(0, 1)
        return foot_t, insole_t


__all__ = [
    "FootInsoleDataset",
    "FootInsoleDatasetConfig",
    "_normalize_points",
    "_resample_points",
    "resample_points_interpolate",
    "normalize_points_with_stats",
    "denormalize_xyz",
]


