from __future__ import annotations

import argparse
import os
import re
import numpy as np
import torch

# 兼容直接脚本运行：将项目根目录加入 sys.path 以支持 `import process.*`
import os as _os
import sys as _sys
_FILE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PROJECT_ROOT = _os.path.dirname(_FILE_DIR)
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

try:
    from .dataset import _normalize_points, _resample_points, resample_points_interpolate, normalize_points_with_stats, denormalize_xyz  # type: ignore
    from .model import DGCNNPointCloud2PointCloud
except Exception:  # 兼容直接脚本运行
    from process.dataset import _normalize_points, _resample_points, resample_points_interpolate, normalize_points_with_stats, denormalize_xyz  # type: ignore
    from process.model import DGCNNPointCloud2PointCloud


_FOOT_PATTERN = re.compile(r"^(\d+)_foot_([LR])\.npy$")


def load_model(checkpoint_path: str, num_points: int, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    k = cfg.get("k", 20)
    emb_dims = cfg.get("emb_dims", 1024)
    dropout = cfg.get("dropout", 0.5)
    input_dims = cfg.get("input_dims", 3)
    # 重要：模型的输出点数固定由训练时的 num_points 决定
    trained_num_points = int(cfg.get("num_points", num_points))
    model = DGCNNPointCloud2PointCloud(k=k, emb_dims=emb_dims, num_points=trained_num_points, dropout=dropout, input_dims=input_dims)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model, int(input_dims), trained_num_points


def infer_single(input_path: str, output_path: str, model, device: torch.device, num_points: int, input_dims: int, target_points: int, secondary_points: int):
    points = np.load(input_path)  # (N, C)
    points, centroid, scale = normalize_points_with_stats(points)
    # 裁剪到与模型一致的输入通道数（例如 3 或 6）
    if points.shape[1] >= input_dims:
        points = points[:, :input_dims]
    points = _resample_points(points, num_points, np.random.RandomState(123))
    x = torch.from_numpy(points.astype(np.float32)).transpose(0, 1).unsqueeze(0).to(device)  # (1, C, N)
    with torch.no_grad():
        pred = model(x)  # (1, 3, N)
    pred_np = pred.squeeze(0).transpose(0, 1).cpu().numpy()  # (N_pred, 3)
    pred_np = denormalize_xyz(pred_np, centroid, scale)
    # 二次采样然后插值回到 target_points
    if target_points and target_points > 0:
        rng = np.random.RandomState(123)
        # 二次采样到 secondary_points（若预测少于 secondary_points，则保持原样）
        sec = min(max(1, secondary_points), target_points)
        if pred_np.shape[0] > sec:
            pred_np = _resample_points(pred_np, sec, rng)
        # 插值上采样/精确采样到 target_points
        if pred_np.shape[0] != target_points:
            if pred_np.shape[0] > target_points:
                pred_np = _resample_points(pred_np, target_points, rng)
            else:
                pred_np = resample_points_interpolate(pred_np, target_points, rng)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, pred_np)
    print(f"保存生成的鞋垫点云到: {output_path}")
    print(f"最终点云点数: {pred_np.shape[0]}")


def infer_batch(test_root: str, feet_subdir: str, output_root: str, output_subdir: str, model_L, model_R, device: torch.device, num_points: int, input_dims: int, target_points: int):
    # 输入目录：优先使用子目录（若存在），否则直接使用 test_root
    candidate_dir = os.path.join(test_root, feet_subdir) if feet_subdir else test_root
    feet_dir = candidate_dir if os.path.isdir(candidate_dir) else test_root

    # 输出目录：若设置了子目录则拼接，否则直接为根目录
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
        pid, side = m.group(1), m.group(2).upper()
        out_name = f"{pid}_insole_{side}.npy"
        out_path = os.path.join(out_dir, out_name)
        model = model_L if side == 'L' else model_R if model_R is not None else model_L
        infer_single(foot_path, out_path, model, device, num_points, input_dims, target_points)
    print("全部完成。")


def main():
    parser = argparse.ArgumentParser(description="Foot point cloud -> Insole point cloud inference (single or batch)")
    # 单文件模式
    parser.add_argument("--input", default=None, help="输入足部点云 .npy 文件 (N, C)")
    parser.add_argument("--output", default=None, help="输出鞋垫点云 .npy 文件 (N, 3)")
    # 批量模式
    parser.add_argument("--test_root", default=os.path.join("test", "pointcloud"), help="测试集根目录（若无子目录，直接放 .npy 文件在此目录下）")
    parser.add_argument("--feet_subdir", default="", help="可选：若测试点云在子目录中（例如 feet），设置此参数；为空表示直接在 test_root 下")
    parser.add_argument("--output_root", default=os.path.join("output", "pointcloud"), help="输出根目录")
    parser.add_argument("--output_subdir", default="insoles", help="可选：输出子目录；为空则直接输出到 output_root 下")
    # 通用
    parser.add_argument("--checkpoint_L", default=os.path.join("checkpoints", "p2p_dgcnn", "models", "best_L.pt"), help="左脚模型权重路径")
    parser.add_argument("--checkpoint_R", default=os.path.join("checkpoints", "p2p_dgcnn", "models", "best_R.pt"), help="右脚模型权重路径（可选；未提供则使用左脚模型）")
    # 编码输入点数：建议保持在 2048~8192，避免 KNN 图构建 OOM
    parser.add_argument("--num_points", type=int, default=4096, help="编码输入重采样点数（建议 2048~8192）")
    parser.add_argument("--target_points", type=int, default=300000, help="最终输出点云点数")
    parser.add_argument("--secondary_points", type=int, default=4096, help="插值前的二次采样点数")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model_L, input_dims, trained_num_points = load_model(args.checkpoint_L, num_points=args.num_points, device=device)
    model_R = None
    if args.checkpoint_R and os.path.exists(args.checkpoint_R):
        try:
            model_R, _, _ = load_model(args.checkpoint_R, num_points=args.num_points, device=device)
        except Exception as _:
            model_R = None

    if args.input is not None and args.output is not None:
        # 根据文件名识别侧别，选择模型
        side = None
        m = _FOOT_PATTERN.match(os.path.basename(args.input))
        if m is not None:
            side = m.group(2).upper()
        mdl = model_L if (side == 'L' or model_R is None) else model_R
        infer_single(args.input, args.output, mdl, device, args.num_points, input_dims, args.target_points, args.secondary_points)
    else:
        # 批处理内部调用单样本接口，保持相同二次采样与插值逻辑
        def _infer_single_wrapper(ip, op):
            side = None
            m = _FOOT_PATTERN.match(os.path.basename(ip))
            if m is not None:
                side = m.group(2).upper()
            mdl = model_L if (side == 'L' or model_R is None) else model_R
            infer_single(ip, op, mdl, device, args.num_points, input_dims, args.target_points, args.secondary_points)
        # 复用现有批处理遍历代码
        # 由于原函数签名不同，这里直接遍历实现
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
                _infer_single_wrapper(foot_path, out_path)
            print("全部完成。")


if __name__ == "__main__":
    main()


