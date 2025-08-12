from __future__ import annotations

import torch
import torch.nn as nn


def pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared distances between points in x and y.

    Args:
        x: (B, N, C)
        y: (B, M, C)
    Returns:
        (B, N, M) distances
    """
    x2 = (x ** 2).sum(-1, keepdim=True)  # (B, N, 1)
    y2 = (y ** 2).sum(-1).unsqueeze(1)   # (B, 1, M)
    xy = x @ y.transpose(1, 2)           # (B, N, M)
    return x2 - 2 * xy + y2


class ChamferDistance(nn.Module):
    """Chamfer Distance between two point sets.

    - Inputs: (B, C, N) and (B, C, M)
    - Output: scalar loss
    - If max_points is set, randomly samples up to that many points from each set per sample
      to bound memory/time (useful for very large point clouds).
    """

    def __init__(self, max_points: int | None = None) -> None:
        super().__init__()
        self.max_points = max_points

    def _maybe_sample(self, pts: torch.Tensor) -> torch.Tensor:
        # pts: (B, N, C)
        if self.max_points is None:
            return pts
        B, N, C = pts.shape
        if N <= self.max_points:
            return pts
        # Per-sample random sampling
        sampled = []
        for b in range(B):
            idx = torch.randperm(N, device=pts.device)[: self.max_points]
            sampled.append(pts[b : b + 1, idx, :])
        return torch.cat(sampled, dim=0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert to (B, N, C)
        pred_t = pred.transpose(1, 2).contiguous()
        targ_t = target.transpose(1, 2).contiguous()

        # Optional downsampling for stability/performance
        pred_t = self._maybe_sample(pred_t)
        targ_t = self._maybe_sample(targ_t)

        dists = pairwise_distances(pred_t, targ_t)  # (B, N, M)

        # For each predicted point, its nearest neighbor in target
        min_pred_to_targ, _ = torch.min(dists, dim=2)  # (B, N)
        # For each target point, its nearest neighbor in predicted
        min_targ_to_pred, _ = torch.min(dists, dim=1)  # (B, M)

        loss = min_pred_to_targ.mean() + min_targ_to_pred.mean()
        return loss


__all__ = ["ChamferDistance"]


