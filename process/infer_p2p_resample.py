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


_FOOT_PATTERN = re.compile(r"^(\d+)_foot_([LR])\.npy$", re.IGNORECASE)


def load_model(checkpoint_path: str, num_points: int, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    k = cfg.get("k", 20)
    emb_dims = cfg.get("emb_dims", 1024)
    dropout = cfg.get("dropout", 0.5)
    input_dims = cfg.get("input_dims", 3)
    trained_num_points = int(cfg.get("gen_points", num_points))
    model = DGCNNPointCloud2PointCloud(k=k, emb_dims=emb_dims, num_points=trained_num_points, dropout=dropout, input_dims=input_dims)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model, int(input_dims)


def infer_single(input_path: str, output_path: str, model, device: torch.device, encode_points: int, input_dims: int, target_points: int):
    points = np.load(input_path)  # (N, C)
    points, centroid, scale = normalize_points_with_stats(points)
    if points.shape[1] >= input_dims:
        points = points[:, :input_dims]
    points = _resample_points(points, encode_points, np.random.RandomState(123))
    x = torch.from_numpy(points.astype(np.float32)).transpose(0, 1).unsqueeze(0).to(device)  # (1, C, N)
    with torch.no_grad():
        pred = model(x)  # (1, 3, N_pred)
    pred_np = pred.squeeze(0).transpose(0, 1).cpu().numpy()  # (N_pred, 3)
    pred_np = denormalize_xyz(pred_np, centroid, scale)
    if target_points and target_points > 0 and pred_np.shape[0] != target_points:
        rng = np.random.RandomState(123)
        if pred_np.shape[0] > target_points:
            pred_np = _resample_points(pred_np, target_points, rng)
        else:
            pred_np = resample_points_interpolate(pred_np, target_points, rng)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, pred_np)
    print(f"保存生成的鞋垫点云到: {output_path}")
    print(f"最终点云点数: {pred_np.shape[0]}")


def infer_batch(test_root: str, feet_subdir: str, output_root: str, output_subdir: str, model, device: torch.device, encode_points: int, input_dims: int, target_points: int, include_all_npy: bool = False, recursive: bool = False):
    candidate_dir = os.path.join(test_root, feet_subdir) if feet_subdir else test_root
    feet_dir = candidate_dir if os.path.isdir(candidate_dir) else test_root
    out_dir = os.path.join(output_root, output_subdir) if output_subdir else output_root
    os.makedirs(out_dir, exist_ok=True)
    # 收集候选文件
    if recursive:
        candidates = []
        for root, _, files in os.walk(feet_dir):
            for f in files:
                if f.lower().endswith('.npy') and (_FOOT_PATTERN.match(f) or include_all_npy):
                    candidates.append(os.path.join(root, f))
    else:
        files = sorted(os.listdir(feet_dir))
        if include_all_npy:
            candidates = [os.path.join(feet_dir, f) for f in files if f.lower().endswith('.npy')]
        else:
            candidates = [os.path.join(feet_dir, f) for f in files if _FOOT_PATTERN.match(f)]
    if not candidates:
        print(f"未在 {feet_dir} 找到可用的 .npy 测试点云文件")
        return
    print(f"共 {len(candidates)} 个样本，开始生成 ...")
    for foot_path in candidates:
        fname = os.path.basename(foot_path)
        m = _FOOT_PATTERN.match(fname)
        if m is not None:
            pid, side = m.group(1), m.group(2).upper()
            out_name = f"{pid}_insole_{side}.npy"
        else:
            base = os.path.splitext(fname)[0]
            out_base = re.sub(r"foot", "insole", base, flags=re.IGNORECASE)
            if out_base == base:
                out_base = base + "_insole"
            out_name = out_base + ".npy"
        out_path = os.path.join(out_dir, out_name)
        infer_single(foot_path, out_path, model, device, encode_points, input_dims, target_points)
    print("全部完成。")


def main():
    parser = argparse.ArgumentParser(description="Inference (resample up to target points)")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--test_root", default=os.path.join("test", "pointcloud"))
    parser.add_argument("--feet_subdir", default="")
    parser.add_argument("--output_root", default=os.path.join("output", "pointcloud"))
    parser.add_argument("--output_subdir", default="insoles")
    parser.add_argument("--checkpoint", default=os.path.join("checkpoints", "p2p_dgcnn", "models", "best.pt"))
    # 编码点数务必保持较小（如 2048/4096），否则 KNN 图构建会 OOM
    parser.add_argument("--encode_points", type=int, default=4096, help="进入模型前的编码点数（建议 2048~8192）")
    parser.add_argument("--target_points", type=int, default=300000, help="最终输出点数")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include_all_npy", action="store_true", help="包含目录下所有 .npy 文件（不限制命名模式）")
    parser.add_argument("--recursive", action="store_true", help="递归搜索子目录中的 .npy 文件")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, input_dims = load_model(args.checkpoint, num_points=args.encode_points, device=device)

    if args.input and args.output:
        infer_single(args.input, args.output, model, device, args.encode_points, input_dims, args.target_points)
    else:
        infer_batch(
            args.test_root,
            args.feet_subdir,
            args.output_root,
            args.output_subdir,
            model,
            device,
            args.encode_points,
            input_dims,
            args.target_points,
            include_all_npy=args.include_all_npy,
            recursive=args.recursive,
        )


if __name__ == "__main__":
    main()


