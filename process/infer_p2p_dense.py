from __future__ import annotations

import argparse
import os
import re
import numpy as np
import torch

# 兼容直接脚本运行
import os as _os
import sys as _sys
_FILE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PROJECT_ROOT = _os.path.dirname(_FILE_DIR)
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

try:
    from .dataset import _resample_points, resample_points_interpolate, normalize_points_with_stats, denormalize_xyz  # type: ignore
    from .model import DGCNNPointCloud2PointCloud
except Exception:  # 兼容直接脚本运行
    from process.dataset import _resample_points, resample_points_interpolate, normalize_points_with_stats, denormalize_xyz  # type: ignore
    from process.model import DGCNNPointCloud2PointCloud


_FOOT_PATTERN = re.compile(r"^(\d+)_foot_([LR])\.npy$")


def load_model_dense(checkpoint_path: str, device: torch.device, input_dims: int = 3, gen_points: int = 300000):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    k = cfg.get("k", 10)
    emb_dims = cfg.get("emb_dims", 1024)
    dropout = cfg.get("dropout", 0.5)
    input_dims = int(cfg.get("input_dims", input_dims))
    # 使用训练时的生成头点数，避免权重尺寸不一致
    trained_num_points = int(cfg.get("gen_points", cfg.get("num_points", 4096)))
    model = DGCNNPointCloud2PointCloud(
        k=k, emb_dims=emb_dims, num_points=trained_num_points, dropout=dropout, input_dims=input_dims
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model, input_dims


def infer_single(input_path: str, output_path: str, model, device: torch.device, encode_points: int, input_dims: int, secondary_points: int):
    points = np.load(input_path)  # (N, C)
    points, centroid, scale = normalize_points_with_stats(points)
    if points.shape[1] >= input_dims:
        points = points[:, :input_dims]
    points = _resample_points(points, encode_points, np.random.RandomState(123))
    x = torch.from_numpy(points.astype(np.float32)).transpose(0, 1).unsqueeze(0).to(device)  # (1, C, N)
    with torch.no_grad():
        pred = model(x)  # (1, 3, 200k)
    pred_np = pred.squeeze(0).transpose(0, 1).cpu().numpy()  # (200k, 3)
    pred_np = denormalize_xyz(pred_np, centroid, scale)
    # 二次采样 + 插值回 300k
    rng = np.random.RandomState(123)
    sec = max(1, secondary_points)
    if pred_np.shape[0] > sec:
        pred_np = _resample_points(pred_np, sec, rng)
    if pred_np.shape[0] != 300000:
        if pred_np.shape[0] > 300000:
            pred_np = _resample_points(pred_np, 300000, rng)
        else:
            pred_np = resample_points_interpolate(pred_np, 300000, rng)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, pred_np)
    print(f"保存生成的鞋垫点云到: {output_path}")
    print(f"最终点云点数: {pred_np.shape[0]}")


def infer_batch(test_root: str, feet_subdir: str, output_root: str, output_subdir: str, model, device: torch.device, encode_points: int, input_dims: int):
    candidate_dir = os.path.join(test_root, feet_subdir) if feet_subdir else test_root
    feet_dir = candidate_dir if os.path.isdir(candidate_dir) else test_root
    out_dir = os.path.join(output_root, output_subdir) if output_subdir else output_root
    os.makedirs(out_dir, exist_ok=True)
    names = [f for f in sorted(os.listdir(feet_dir)) if _FOOT_PATTERN.match(f)]
    if not names:
        print(f"未在 {feet_dir} 找到测试点云 (foot) 文件")
        return
    print(f"共 {len(names)} 个样本，开始生成 ...")
    for fname in names:
        foot_path = os.path.join(feet_dir, fname)
        m = _FOOT_PATTERN.match(fname)
        assert m is not None
        pid, side = m.group(1), m.group(2)
        out_name = f"{pid}_insole_{side}.npy"
        out_path = os.path.join(out_dir, out_name)
        infer_single(foot_path, out_path, model, device, encode_points, input_dims)
    print("全部完成。")


def main():
    parser = argparse.ArgumentParser(description="Inference (dense generator 300k with secondary resample+interpolate)")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--test_root", default=os.path.join("test", "pointcloud"))
    parser.add_argument("--feet_subdir", default="")
    parser.add_argument("--output_root", default=os.path.join("output", "pointcloud"))
    parser.add_argument("--output_subdir", default="")
    parser.add_argument("--checkpoint", default=os.path.join("checkpoints", "p2p_dgcnn", "models", "best.pt"))
    parser.add_argument("--encode_points", type=int, default=4096)
    parser.add_argument("--secondary_points", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, input_dims = load_model_dense(args.checkpoint, device=device, gen_points=300000)

    if args.input and args.output:
        infer_single(args.input, args.output, model, device, args.encode_points, input_dims, args.secondary_points)
    else:
        # 复用批量遍历，但在单样本调用中执行二次采样与插值逻辑
        candidate_dir = os.path.join(args.test_root, args.feet_subdir) if args.feet_subdir else args.test_root
        feet_dir = candidate_dir if os.path.isdir(candidate_dir) else args.test_root
        out_dir = os.path.join(args.output_root, args.output_subdir) if args.output_subdir else args.output_root
        os.makedirs(out_dir, exist_ok=True)
        names = [f for f in sorted(os.listdir(feet_dir)) if _FOOT_PATTERN.match(f)]
        if not names:
            print(f"未在 {feet_dir} 找到测试点云 (foot) 文件")
        else:
            print(f"共 {len(names)} 个样本，开始生成 ...")
            for fname in names:
                foot_path = os.path.join(feet_dir, fname)
                m = _FOOT_PATTERN.match(fname)
                assert m is not None
                pid, side = m.group(1), m.group(2)
                out_name = f"{pid}_insole_{side}.npy"
                out_path = os.path.join(out_dir, out_name)
                infer_single(foot_path, out_path, model, device, args.encode_points, input_dims, args.secondary_points)
            print("全部完成。")


if __name__ == "__main__":
    main()


