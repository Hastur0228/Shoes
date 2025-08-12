from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader

# 兼容直接脚本运行：将项目根目录加入 sys.path 以支持 `import process.*`
import os as _os
import sys as _sys
_FILE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PROJECT_ROOT = _os.path.dirname(_FILE_DIR)
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

try:
    from .dataset import FootInsoleDataset, FootInsoleDatasetConfig
    from .losses import ChamferDistance
    from .model import DGCNNPointCloud2PointCloud
    from .util import IOStream, set_random_seeds
except Exception:  # 兼容直接脚本运行
    from process.dataset import FootInsoleDataset, FootInsoleDatasetConfig
    from process.losses import ChamferDistance
    from process.model import DGCNNPointCloud2PointCloud
    from process.util import IOStream, set_random_seeds


@dataclass
class TrainConfig:
    exp_name: str = "p2p_dgcnn"
    data_root: str = os.path.join("data", "pointcloud")
    # 训练数据重采样点数（最终目标：300,000 点）
    num_points: int = 300000
    # 编码阶段二次采样到较小点数以避免显存爆炸
    encode_points: int = 4096
    # 生成头输出点数（较小以控显存；推理阶段再上采样到更高密度）
    gen_points: int = 4096
    batch_size: int = 1
    test_batch_size: int = 1
    epochs: int = 50
    lr: float = 1e-3
    momentum: float = 0.9
    use_sgd: bool = False
    # 大规模点云下邻居不宜过大
    k: int = 10
    emb_dims: int = 1024
    dropout: float = 0.5
    # 大单样本IO与稳定性
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = os.path.join("checkpoints")
    save_every: int = 5
    seed: int = 1
    # 归一化记录（用于推理还原）
    save_norm_stats: bool = True
    # 提前停止
    early_stop_patience: int = 20
    early_stop_min_delta: float = 1e-4
    # 侧别训练: 'L' 仅左脚, 'R' 仅右脚, None 使用两侧
    side_filter: str | None = None
    # 训练期间：将模型预测通过“二次采样→插值”恢复到该点数
    final_target_points: int = 300000
    secondary_points: int = 4096
    # Accuracy 指标（基于最近邻半径阈值）
    acc_threshold: float = 0.02  # 规范化到单位球坐标系下的阈值
    acc_max_points: int = 8192   # 计算精度时参与最近邻的最大点数（降采样以控显存）


def create_dataloaders(cfg: TrainConfig):
    ds_cfg = FootInsoleDatasetConfig(
        root_dir=cfg.data_root,
        num_points=cfg.num_points,
        use_normals=False,
        side_filter=cfg.side_filter,
    )
    train_ds = FootInsoleDataset(ds_cfg, split=(0.9, 0.1), is_train=True)
    val_ds = FootInsoleDataset(ds_cfg, split=(0.9, 0.1), is_train=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.test_batch_size, shuffle=False, num_workers=cfg.num_workers)
    return train_loader, val_loader, ds_cfg


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for foot, insole in loader:
            foot = foot.to(device)
            insole = insole.to(device)
            pred = model(foot)
            # 验证同样执行“二次采样→插值到目标点数”，以与训练一致
            pred = _postprocess_points_secondary_then_interpolate(pred, secondary_points=4096, target_points=300000)
            loss = criterion(pred, insole)
            total_loss += loss.item() * foot.size(0)
            count += foot.size(0)
    return total_loss / max(1, count)


def _postprocess_points_secondary_then_interpolate(points_b3n: torch.Tensor, secondary_points: int, target_points: int) -> torch.Tensor:
    """将 (B, 3, N) 点云先二次采样到 secondary_points，再插值回 target_points。

    注意：仅对 XYZ 进行线性插值。该步骤在张量所在设备上执行。
    """
    if target_points <= 0:
        return points_b3n
    b, c, n = points_b3n.shape
    assert c == 3, "生成头输出应为 3 维 XYZ"
    x = points_b3n
    # 二次采样（如果需要）
    sec = max(1, min(secondary_points, target_points))
    if n > sec:
        idx = torch.randperm(n, device=x.device)[:sec]
        x = x[:, :, idx]
        n = sec
    # 精确到 target_points
    if n == target_points:
        return x
    if n > target_points:
        idx = torch.randperm(n, device=x.device)[:target_points]
        return x[:, :, idx]
    # n < target_points: 线性插值补点
    need = target_points - n
    # 对每个 batch 独立生成插值点
    new_chunks = []
    for bi in range(b):
        xb = x[bi]  # (3, n)
        if n == 1:
            # 只有一个点时，复制
            rep = xb.unsqueeze(2).repeat(1, 1, need)
            new_chunks.append(rep)
            continue
        idx_a = torch.randint(0, n, (need,), device=x.device)
        idx_b = torch.randint(0, n, (need,), device=x.device)
        pa = xb[:, idx_a]  # (3, need)
        pb = xb[:, idx_b]  # (3, need)
        alpha = torch.rand(1, need, device=x.device)
        newb = alpha * pa + (1.0 - alpha) * pb  # (3, need)
        new_chunks.append(newb)
    new_pts = torch.stack(new_chunks, dim=0)  # (B, 3, need)
    out = torch.cat([x, new_pts], dim=2)  # (B, 3, target)
    return out


def train(cfg: Optional[TrainConfig] = None):
    cfg = cfg or TrainConfig()
    exp_dir = os.path.join(cfg.out_dir, cfg.exp_name)
    model_dir = os.path.join(exp_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    log_path = os.path.join(exp_dir, "run.log")
    open(log_path, "w", encoding="utf-8").close()
    io = IOStream(log_path)
    io.cprint(str(cfg))

    set_random_seeds(cfg.seed, cuda=(cfg.device == "cuda"))

    train_loader, val_loader, ds_cfg = create_dataloaders(cfg)
    device = torch.device(cfg.device)
    # 根据数据集配置决定输入通道数
    input_dims = 6 if ds_cfg.use_normals else 3
    # 注意：编码输入使用 cfg.num_points；生成头采用较小的 cfg.gen_points
    model = DGCNNPointCloud2PointCloud(
        k=cfg.k,
        emb_dims=cfg.emb_dims,
        num_points=cfg.gen_points,
        dropout=cfg.dropout,
        input_dims=input_dims,
    ).to(device)
    # 记录输入通道数到配置，便于推理保持一致
    cfg.input_dims = input_dims  # type: ignore[attr-defined]
    # 记录点数设定，便于推理端读取
    cfg.trained_input_points = cfg.num_points  # type: ignore[attr-defined]
    cfg.gen_points = cfg.gen_points  # type: ignore[attr-defined]
    if torch.cuda.device_count() > 1 and cfg.device == "cuda":
        model = torch.nn.DataParallel(model)
        io.cprint(f"使用 {torch.cuda.device_count()} 张 GPU 进行训练")

    # 为避免 O(N^2) 显存爆炸，损失计算对两侧各随机采样最多 8192 点
    criterion = ChamferDistance(max_points=8192)
    if cfg.use_sgd:
        io.cprint("优化器: SGD")
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr * 100, momentum=cfg.momentum, weight_decay=1e-4)
    else:
        io.cprint("优化器: Adam")
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs, eta_min=cfg.lr)

    best_val = float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for foot, insole in train_loader:
            foot = foot.to(device)
            insole = insole.to(device)
            # 二次采样编码输入
            if foot.size(-1) > cfg.encode_points:
                idx = torch.randperm(foot.size(-1), device=foot.device)[: cfg.encode_points]
                foot = foot.index_select(dim=-1, index=idx)

            optimizer.zero_grad(set_to_none=True)
            pred = model(foot)
            # 训练时将预测通过“二次采样→插值”恢复到 300k 点
            pred = _postprocess_points_secondary_then_interpolate(pred, secondary_points=cfg.secondary_points, target_points=cfg.final_target_points)
            loss = criterion(pred, insole)
            loss.backward()
            optimizer.step()

            running += loss.item() * foot.size(0)
            seen += foot.size(0)

        train_loss = running / max(1, seen)
        # 验证时同样进行二次采样
        def _validate_with_encode():
            model.eval()
            total_loss = 0.0
            total_count = 0
            with torch.no_grad():
                for f, ins in val_loader:
                    f = f.to(device)
                    ins = ins.to(device)
                    if f.size(-1) > cfg.encode_points:
                        vidx = torch.randperm(f.size(-1), device=f.device)[: cfg.encode_points]
                        f = f.index_select(dim=-1, index=vidx)
                    p = model(f)
                    # 验证时同样恢复到最终点数以与训练/推理一致
                    p = _postprocess_points_secondary_then_interpolate(
                        p, secondary_points=cfg.secondary_points, target_points=cfg.final_target_points
                    )
                    L = criterion(p, ins)
                    total_loss += L.item() * f.size(0)
                    total_count += f.size(0)
            return total_loss / max(1, total_count)

        val_loss = _validate_with_encode()
        
        # 计算 Accuracy（基于阈值的最近邻命中率），在规范化坐标系下进行
        def _compute_accuracy_nn_threshold() -> float:
            model.eval()
            import torch as _torch
            thr = float(cfg.acc_threshold)
            max_pts = int(cfg.acc_max_points)
            total_hit = 0
            total_cnt = 0
            with _torch.no_grad():
                for f, ins in val_loader:
                    f = f.to(device)
                    ins = ins.to(device)
                    # 与训练一致的编码点数处理
                    if f.size(-1) > cfg.encode_points:
                        vidx = _torch.randperm(f.size(-1), device=f.device)[: cfg.encode_points]
                        f = f.index_select(dim=-1, index=vidx)
                    p = model(f)  # (B, 3, n)
                    # 与训练相同的“二次采样→插值”恢复到最终点数
                    p = _postprocess_points_secondary_then_interpolate(
                        p, secondary_points=cfg.secondary_points, target_points=cfg.final_target_points
                    )  # (B, 3, N)
                    # 转为 (B, N, C)
                    pred_bn3 = p.transpose(1, 2).contiguous()
                    targ_bn3 = ins.transpose(1, 2).contiguous()
                    # 限制点数以控显存
                    def _maybe_sample(x: _torch.Tensor) -> _torch.Tensor:
                        b, n, c = x.shape
                        if n <= max_pts:
                            return x
                        idx = _torch.randperm(n, device=x.device)[: max_pts]
                        return x[:, idx, :]
                    pred_bn3 = _maybe_sample(pred_bn3)
                    targ_bn3 = _maybe_sample(targ_bn3)
                    # 最近邻距离（平方）并取最小
                    # 使用项目内 pairwise_distances 的等价逻辑，避免额外导入
                    x2 = (pred_bn3 ** 2).sum(-1, keepdim=True)        # (B, N, 1)
                    y2 = (targ_bn3 ** 2).sum(-1).unsqueeze(1)         # (B, 1, M)
                    xy = pred_bn3 @ targ_bn3.transpose(1, 2)          # (B, N, M)
                    d2 = x2 - 2 * xy + y2                              # (B, N, M)
                    min_d2, _ = _torch.min(d2, dim=2)                  # (B, N)
                    # 阈值基于欧式距离
                    hits = (min_d2.sqrt() <= thr).sum().item()
                    cnt = min_d2.numel()
                    total_hit += hits
                    total_cnt += cnt
            return float(total_hit) / max(1, total_cnt)
        acc_val = _compute_accuracy_nn_threshold()
        # 在完成本 epoch 的优化步骤后再更新调度器，避免 PyTorch 的顺序警告
        scheduler.step()

        io.cprint(
            f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f} | acc {acc_val:.4f} | lr {scheduler.get_last_lr()[0]:.2e}"
        )

        # 早停与最优保存
        if (best_val - val_loss) > cfg.early_stop_min_delta:
            best_val = val_loss
            best_epoch = epoch
            no_improve_epochs = 0
            best_path = os.path.join(model_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "val_loss": val_loss,
            }, best_path)
            io.cprint(f"保存最佳权重: epoch={epoch}, val_loss={val_loss:.6f}, 路径={best_path}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= cfg.early_stop_patience:
                io.cprint(f"提前停止: 连续 {no_improve_epochs} 个 epoch 无显著提升 (best @ epoch {best_epoch}, best_val={best_val:.6f})")
                break

        if epoch % cfg.save_every == 0:
            ckpt_path = os.path.join(model_dir, f"epoch_{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "val_loss": val_loss,
            }, ckpt_path)

    io.close()


if __name__ == "__main__":
    train()


