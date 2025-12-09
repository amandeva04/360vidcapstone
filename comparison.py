import argparse
import glob
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import load_and_resample, make_windows
from models import GazeGRU
from training import NpSeqDataset
from lossfunction import cosine_loss, _l2norm, _safe_dot

# 🔧 hardcode the models you want to compare
CHECKPOINT_PATHS = [
    "Trained_Models/latest.pt",
    "Trained_Models/LSTM.pt",
    "Trained_Models/kalman.pt",
    #"Trained_Models/TCN.pt",
    # "Trained_Models/your_other_model.pt",
]

def parse_args():
    ap = argparse.ArgumentParser(
        "Compare multiple trained GazeGRU checkpoints using their own data configs"
    )
    ap.add_argument(
        "--glob",
        default="Formated_Data/Experiment_1/**/video_*.csv",
        help="Pattern for CSVs (relative or absolute); ** is recursive",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=256,
        help="batch size for evaluation",
    )
    ap.add_argument(
        "--device",
        default=(
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        ),
        help="device to run evaluation on",
    )
    ap.add_argument(
        "--hit_thresholds",
        type=float,
        nargs="*",
        default=[5.0, 10.0],
        help="Angular error thresholds in degrees for hit-rate metrics",
    )
    return ap.parse_args()


def split_list(xs, fracs=(0.7, 0.15, 0.15)):
    """Same split logic as in main.py: [0:train, train:val, val:] by video."""
    n = len(xs)
    a = int(fracs[0] * n)
    b = int((fracs[0] + fracs[1]) * n)
    return xs[:a], xs[a:b], xs[b:]


def stack_windows(seq_list, T, horizon, add_xyz_vel):
    """Apply make_windows to each dataframe and stack the resulting arrays."""
    Xs, Ys = [], []
    for df in seq_list:
        X, Y = make_windows(df, T=T, horizon=horizon, add_xyz_vel=add_xyz_vel)
        if len(X):
            Xs.append(X)
            Ys.append(Y)
    if not Xs:
        feat_dim = 8 + (3 if add_xyz_vel else 0)
        return (
            np.zeros((0, T, feat_dim), np.float32),
            np.zeros((0, 3), np.float32),
        )
    return np.concatenate(Xs), np.concatenate(Ys)


def angular_error_deg_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Per-sample angular error in degrees between 3D vectors.

    pred, target: (..., 3)
    Returns: tensor of shape (...) with angle in degrees.
    """
    pred_n = _l2norm(pred)
    target_n = _l2norm(target)
    dot = _safe_dot(pred_n, target_n)
    ang = torch.acos(dot)  # radians
    ang = torch.nan_to_num(ang, nan=0.0)
    return ang * (180.0 / math.pi)


def evaluate_model_on_loader(
    model: torch.nn.Module,
    dl: DataLoader,
    device: str,
    hit_thresholds: List[float],
) -> Dict[str, float]:
    """
    Compute several metrics for a model on a given DataLoader.

    Metrics:
      - mean_cosine_loss: same objective used during training (lower is better)
      - mean_angular_deg: mean angular error in degrees (lower is better)
      - median_angular_deg: robust central tendency of angular error
      - rmse_vec: RMSE on the 3D direction vector components
      - hit@Xdeg: fraction of samples with angular error <= X degrees
    """
    model.to(device)
    model.eval()

    total_n = 0
    sum_cos = 0.0
    sum_ang = 0.0
    sum_sq_err = 0.0  # for RMSE over vector components
    all_deg: List[float] = []
    hit_counts = np.zeros(len(hit_thresholds), dtype=np.int64)

    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            B = xb.shape[0]
            if B == 0:
                continue

            pred, _ = model(xb)  # (B, 3)

            # Cosine loss (mean over batch)
            batch_cos = cosine_loss(pred, yb).item()

            # Per-sample angular error (deg)
            deg = angular_error_deg_per_sample(pred, yb)  # (B,)
            # Vector error for RMSE
            err_vec = pred - yb
            batch_sq_err = (err_vec ** 2).sum().item()  # sum over B and 3

            # Update aggregates
            total_n += B
            sum_cos += batch_cos * B
            sum_ang += deg.sum().item()
            sum_sq_err += batch_sq_err
            all_deg.extend(deg.cpu().tolist())

            for j, th in enumerate(hit_thresholds):
                hit_counts[j] += (deg <= th).sum().item()

    if total_n == 0:
        return {
            "mean_cosine_loss": float("nan"),
            "mean_angular_deg": float("nan"),
            "median_angular_deg": float("nan"),
            "rmse_vec": float("nan"),
            **{f"hit@{th:g}deg": float("nan") for th in hit_thresholds},
        }

    mean_cos = sum_cos / total_n
    mean_ang = sum_ang / total_n
    median_ang = float(np.median(np.asarray(all_deg, dtype=np.float32)))
    rmse = math.sqrt(sum_sq_err / (total_n * 3.0))

    metrics: Dict[str, float] = {
        "mean_cosine_loss": mean_cos,
        "mean_angular_deg": mean_ang,
        "median_angular_deg": median_ang,
        "rmse_vec": rmse,
    }
    for j, th in enumerate(hit_thresholds):
        metrics[f"hit@{th:g}deg"] = hit_counts[j] / float(total_n)

    return metrics


def main():
    args = parse_args()
    ROOT = Path(__file__).resolve().parent

    # ---- Find all CSV files once ----
    pat_path = Path(args.glob)
    pattern = (
        str((ROOT / pat_path).resolve())
        if not pat_path.is_absolute()
        else str(pat_path)
    )

    files = sorted(
        set(glob.glob(pattern, recursive=True))
        | set(glob.glob(pattern.replace(".csv", ".CSV"), recursive=True))
    )
    print(f"[INFO] comparison script dir = {ROOT}")
    print(f"[INFO] pattern              = {pattern}")
    print(f"[INFO] matched files        = {len(files)}")
    if not files:
        raise SystemExit("No CSV files found; check --glob or folder structure.")

    if not CHECKPOINT_PATHS:
        print("[ERROR] CHECKPOINT_PATHS is empty. Edit comparison.py and add some .pt paths.")
        return

    # optional cache so we don't recompute windows for identical configs
    data_cache = {}

    results: List[Tuple[str, Dict[str, float]]] = []

    for ckpt_str in CHECKPOINT_PATHS:
        ckpt_path = (ROOT / ckpt_str).resolve()
        if not ckpt_path.is_file():
            print(f"[WARN] checkpoint not found: {ckpt_path}")
            continue

        print(f"\n[MODEL] {ckpt_path}")
        blob = torch.load(str(ckpt_path), map_location="cpu")
        cfg = blob.get("config", {}) or {}

        # Pull data-related configs from the checkpoint, with safe defaults
        hz = float(cfg.get("hz", 60.0))
        T = int(cfg.get("T", 20))
        horizon = int(cfg.get("horizon", 1))
        add_xyz_vel = bool(cfg.get("add_xyz_vel", 0))
        zscore = int(cfg.get("zscore", 0))

        hidden = int(cfg.get("hidden", 96))
        layers = int(cfg.get("layers", 1))

        print(f"[INFO] using cfg from checkpoint: hz={hz}, T={T}, horizon={horizon}, "
              f"add_xyz_vel={add_xyz_vel}, zscore={zscore}, hidden={hidden}, layers={layers}")

        # ---- Build / reuse windowed data for this config ----
        cfg_key = (hz, T, horizon, add_xyz_vel, zscore)

        if cfg_key in data_cache:
            Xte, Yte, in_dim = data_cache[cfg_key]
            print("[INFO] reusing cached test windows for this config")
        else:
            # 1) Load & resample per file according to this cfg
            seqs = []
            need_len = T + horizon
            for fp in files:
                df = load_and_resample(fp, hz=hz)
                if len(df) >= need_len:
                    seqs.append(df)
            print(f"[INFO] usable sequences (>= {need_len} frames): {len(seqs)}")
            if not seqs:
                print("[WARN] no usable sequences for this config; skipping model.")
                continue

            # 2) Split into train/val/test as in main.py
            train_seqs, val_seqs, test_seqs = split_list(seqs, (0.7, 0.15, 0.15))

            # 3) Window & stack
            Xtr, Ytr = stack_windows(train_seqs, T, horizon, add_xyz_vel)
            Xva, Yva = stack_windows(val_seqs,   T, horizon, add_xyz_vel)
            Xte_raw, Yte = stack_windows(test_seqs,  T, horizon, add_xyz_vel)
            print(f"[INFO] windows — train:{len(Xtr)}  val:{len(Xva)}  test:{len(Xte_raw)}")
            if not len(Xte_raw):
                print("[WARN] test split has no windows for this config; skipping model.")
                continue

            # 4) Optional z-score using train stats (mirror main.py)
            if zscore and len(Xtr) > 0:
                mu = Xtr.mean(axis=(0, 1), keepdims=True)
                sigma = Xtr.std(axis=(0, 1), keepdims=True).clip(1e-6)
                Xte = (Xte_raw - mu) / sigma
                print("[INFO] applied z-score standardization (train stats) to test features")
            else:
                Xte = Xte_raw

            in_dim = int(Xte.shape[-1])
            print(f"[INFO] inferred in_dim for this config = {in_dim}")

            # store in cache
            data_cache[cfg_key] = (Xte, Yte, in_dim)

        # ---- Build DataLoader for this model ----
        dl_te = DataLoader(NpSeqDataset(Xte, Yte), batch_size=args.batch, shuffle=False)

        # ---- Build model and load weights ----
        model = GazeGRU(in_dim=in_dim, hidden=hidden, layers=layers)
        model.load_state_dict(blob["model_state"])

        # ---- Evaluate ----
        metrics = evaluate_model_on_loader(
            model, dl_te, device=args.device, hit_thresholds=args.hit_thresholds
        )
        results.append((ckpt_path.name, metrics))

    if not results:
        print("[INFO] No valid checkpoints were evaluated.")
        return

    # ---- Pretty-print a comparison table ----
    metric_names = list(results[0][1].keys())

    print("\n=== Model comparison (each model evaluated with its own cfg) ===")
    header = ["model"] + metric_names
    row_fmt = "{:<30} " + " ".join(["{:>15}"] * len(metric_names))
    print(row_fmt.format(*header))
    print("-" * (30 + 16 * len(metric_names)))
    for name, metrics in results:
        vals = [metrics[m] for m in metric_names]
        nice_vals = [
            f"{v:.4f}" if isinstance(v, (float, int)) and not math.isnan(float(v)) else "nan"
            for v in vals
        ]
        print(row_fmt.format(name, *nice_vals))


if __name__ == "__main__":
    main()
