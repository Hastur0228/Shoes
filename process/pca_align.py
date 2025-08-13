from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class PCAAlignConfig:
    """
    PCA 对齐与归一化配置（面向足模/鞋垫点云）。

    - toe_positive_x:     是否强制脚尖指向 +X（通过端部宽度启发式判断长轴方向）
    - origin_mode:        原点统一方式：'heel'（脚跟切片质心，Z 取最小值）或 'bottom'（底部点云质心，Z 取最小值）
    - heel_slice_ratio:   计算脚跟切片所占长度比例（0-0.2 合理；默认 8%）
    - scale_mode:         尺寸归一化方式：'none' | 'length' | 'width'
    - target_length:      当 scale_mode='length' 时，目标脚长（与输入单位一致，如毫米）
    - target_width:       当 scale_mode='width' 时，目标最大宽度
    """

    toe_positive_x: bool = True
    origin_mode: str = "heel"  # 'heel' | 'bottom'
    heel_slice_ratio: float = 0.08
    scale_mode: str = "none"  # 'none' | 'length' | 'width'
    target_length: float = 1.0
    target_width: float = 1.0


@dataclass
class PCAAlignStats:
    """返回的对齐统计量与变换参数。"""

    centroid: np.ndarray          # (3,) PCA 前的几何中心
    rotation: np.ndarray          # (3, 3) 旋转矩阵（PCA 基 -> XYZ）
    sign_flips: np.ndarray        # (3,) 每轴符号翻转（+1/-1）
    translation: np.ndarray       # (3,) 平移向量（对齐后减去的原点）
    scale: float                  # 尺度缩放（1 表示未缩放）
    length_xyz: np.ndarray        # (3,) 对齐后包围盒尺寸 (Lx, Ly, Lz)


def _compute_pca_axes(points_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算点云的 PCA 轴。

    返回:
        centroid: (3,) 质心
        evecs:    (3, 3) 列向量为主轴，按特征值从大到小排序
    """
    assert points_xyz.ndim == 2 and points_xyz.shape[1] == 3
    centroid = points_xyz.mean(axis=0)
    centered = points_xyz - centroid
    cov = centered.T @ centered / max(1, centered.shape[0] - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    return centroid, evecs


def _align_to_canonical_axes(points_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将点云通过 PCA 对齐到 (X=长轴, Y=宽轴, Z=高轴)。

    返回:
        points_aligned: (N, 3) 对齐后的坐标
        R:              (3, 3) 旋转矩阵，使得 x' = R @ (x - centroid)
        centroid:       (3,)   原始质心
    """
    centroid, evecs = _compute_pca_axes(points_xyz)
    # 将 PCA 基底映射到世界坐标：R = E^T
    R = evecs.T
    points_aligned = (points_xyz - centroid) @ R.T  # 等价于 R @ (x-centroid)
    return points_aligned, R, centroid


def _resolve_long_axis_sign_by_width(points_aligned: np.ndarray, ratio: float) -> int:
    """
    通过端部宽度（Y 方向包围盒）启发式确定长轴 X 的正向：
    - 取靠近 x_min 与 x_max 的两个切片（各占总长的 ratio），比较其在 Y 方向的宽度。
    - 通常脚尖端更宽，若 x_max 侧更宽，则 +X 指向脚尖；否则取反。

    返回:
        sgn_x: +1 或 -1
    """
    x = points_aligned[:, 0]
    y = points_aligned[:, 1]
    x_min, x_max = float(x.min()), float(x.max())
    total_len = max(1e-9, x_max - x_min)
    slice_len = total_len * float(np.clip(ratio, 1e-4, 0.3))

    left_mask = x <= (x_min + slice_len)
    right_mask = x >= (x_max - slice_len)

    def _width(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        yy = y[mask]
        return float(yy.max() - yy.min()) if yy.size > 0 else 0.0

    w_left = _width(left_mask)
    w_right = _width(right_mask)
    return +1 if w_right >= w_left else -1


def _choose_origin(points_aligned: np.ndarray, mode: str, heel_ratio: float) -> np.ndarray:
    """
    计算统一原点：
    - 'heel':   X 最小端的一段切片的 (x,y) 质心，Z 取全局最小（地面）
    - 'bottom': 底部（Z 最小附近）点云的几何质心，Z 取全局最小
    返回 (3,) 平移向量（将该点移到原点）。
    """
    p = points_aligned
    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]
    z0 = float(z.min())

    mode = mode.lower().strip()
    if mode == "heel":
        x_min, x_max = float(x.min()), float(x.max())
        total_len = max(1e-9, x_max - x_min)
        slice_len = total_len * float(np.clip(heel_ratio, 1e-4, 0.3))
        heel_mask = x <= (x_min + slice_len)
        if np.any(heel_mask):
            xy_centroid = np.array([x[heel_mask].mean(), y[heel_mask].mean()], dtype=np.float64)
        else:
            xy_centroid = np.array([x_min, float(y.mean())], dtype=np.float64)
        origin = np.array([xy_centroid[0], xy_centroid[1], z0], dtype=np.float64)
        return origin
    elif mode == "bottom":
        # 取靠近 z_min 的薄层点云的质心
        z_min, z_max = float(z.min()), float(z.max())
        thickness = max(1e-9, (z_max - z_min) * 0.02)
        bottom_mask = z <= (z_min + thickness)
        if np.any(bottom_mask):
            bot_centroid = p[bottom_mask].mean(axis=0)
            return np.array([bot_centroid[0], bot_centroid[1], z0], dtype=np.float64)
        return np.array([float(x.mean()), float(y.mean()), z0], dtype=np.float64)
    else:
        raise ValueError("origin_mode 只能是 'heel' 或 'bottom'")


def _compute_scale(points_aligned: np.ndarray, mode: str, target_len: float, target_wid: float) -> float:
    """根据选择的尺度模式计算缩放因子。"""
    mode = mode.lower().strip()
    if mode == "none":
        return 1.0
    x = points_aligned[:, 0]
    y = points_aligned[:, 1]
    if mode == "length":
        cur = float(x.max() - x.min())
        return float(target_len) / max(1e-9, cur)
    if mode == "width":
        cur = float(y.max() - y.min())
        return float(target_wid) / max(1e-9, cur)
    raise ValueError("scale_mode 只能是 'none' | 'length' | 'width'")


def pca_align_and_normalize(points: np.ndarray, config: PCAAlignConfig) -> Tuple[np.ndarray, PCAAlignStats]:
    """
    对点云执行 PCA 对齐、方向统一、原点统一与可选尺度归一化。

    输入:
        points: (N, C) 点云，至少包含前 3 列 XYZ；如包含法向量，则形状应为 (N, 6: xyz+normal)
        config: 对齐/归一化配置

    返回:
        new_points: (N, C) 变换后的点云（若包含法向，法向也会旋转/翻转并单位化）
        stats:      变换参数（可用于记录/复现）
    """
    assert points.ndim == 2 and points.shape[1] >= 3
    has_normals = points.shape[1] >= 6

    xyz = points[:, :3].astype(np.float64, copy=True)
    pts_aligned, R, centroid = _align_to_canonical_axes(xyz)

    # 方向统一：长轴朝向脚尖（+X）。
    sgn_x = 1
    if config.toe_positive_x:
        sgn_x = _resolve_long_axis_sign_by_width(pts_aligned, config.heel_slice_ratio)

    # 保持右手系：这里不强制翻转 Y/Z，只对 X 使用启发式符号。
    sign_flips = np.array([sgn_x, 1, 1], dtype=np.float64)

    # 应用符号翻转
    pts_aligned = pts_aligned * sign_flips[None, :]

    # 计算原点（heel/bottom）并平移
    origin = _choose_origin(pts_aligned, config.origin_mode, config.heel_slice_ratio)
    pts_t = pts_aligned - origin[None, :]

    # 可选尺度归一化
    scale = _compute_scale(pts_t, config.scale_mode, config.target_length, config.target_width)
    pts_s = pts_t * float(scale)

    # 对齐后的包围盒尺寸
    min_xyz = pts_s.min(axis=0)
    max_xyz = pts_s.max(axis=0)
    length_xyz = (max_xyz - min_xyz).astype(np.float64)

    out = points.copy().astype(np.float64)
    out[:, :3] = pts_s

    # 法向：仅旋转/符号翻转（不平移、不缩放），并单位化
    if has_normals:
        n = points[:, 3:6].astype(np.float64, copy=True)
        # 与几何相同的旋转 R 和符号翻转（相当于对坐标轴方向翻转）
        n_rot = (n @ R.T) * sign_flips[None, :]
        # 缩放不影响方向，单位化以防数值误差
        norms = np.linalg.norm(n_rot, axis=1, keepdims=True) + 1e-12
        out[:, 3:6] = n_rot / norms

    stats = PCAAlignStats(
        centroid=centroid.astype(np.float64),
        rotation=R.astype(np.float64),
        sign_flips=sign_flips,
        translation=origin.astype(np.float64),
        scale=float(scale),
        length_xyz=length_xyz,
    )

    return out.astype(points.dtype, copy=False), stats


__all__ = [
    "PCAAlignConfig",
    "PCAAlignStats",
    "pca_align_and_normalize",
]


