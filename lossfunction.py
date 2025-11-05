# lossfunction.py
import math
import torch

_EPS = 1e-7  # tight clamp margin for acos stability

def _l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

def _safe_dot(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # u, v: (..., 3) (ideally unit vectors)
    dot = (u * v).sum(dim=-1)
    dot = torch.nan_to_num(dot, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(dot, -1.0 + _EPS, 1.0 - _EPS)

def angular_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean angular distance in radians between predicted and target 3D unit vectors.
    Stable to NaNs and acos domain issues.
    """
    # Extra safety: renormalize in case upstream had drift
    pred = _l2norm(pred)
    target = _l2norm(target)
    dot = _safe_dot(pred, target)
    ang = torch.acos(dot)              # radians
    ang = torch.nan_to_num(ang, nan=0.0)
    return ang.mean()

@torch.no_grad()
def angular_error_deg(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = _l2norm(pred)
    target = _l2norm(target)
    dot = _safe_dot(pred, target)
    ang = torch.acos(dot)
    ang = torch.nan_to_num(ang, nan=0.0)
    return (ang * (180.0 / math.pi)).mean().item()
