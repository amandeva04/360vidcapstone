# lossfunction.py
import math
import torch

_EPS = 1e-7  # tight clamp margin for acos stability

def _l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

def _safe_dot(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # u, v: (..., 3)
    dot = (u * v).sum(dim=-1)
    dot = torch.nan_to_num(dot, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(dot, -1.0 + _EPS, 1.0 - _EPS)

# ---- New: smoother, gradient-friendly training loss ----
def cosine_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    (1 - cosine similarity) averaged. Does NOT pre-normalize pred,
    so early training (when ||pred|| ~ 0) still gets usable gradients.
    """
    # norms (safe)
    pn = pred.norm(dim=-1).clamp_min(eps)
    tn = target.norm(dim=-1).clamp_min(eps)
    # cosine similarity
    cos = (pred * target).sum(dim=-1) / (pn * tn)
    cos = torch.clamp(cos, -1.0 + _EPS, 1.0 - _EPS)
    return (1.0 - cos).mean()

# Keep for eval or if you want to compare
def angular_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
