import argparse
import glob
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import load_and_resample, make_windows
from models import GazeTCN
from training import NpSeqDataset

EPS = 1e-7


def parse_args():
    ap = argparse.ArgumentParser("TCN model comparison on test set")
    ap.add_argument(
        "--ckpt_glob",
        default="Trained_Models/*.pt",
        help="glob pattern for checkpoints (e.g. Trained_Models/*.pt)",
    )
    ap.add_argument(
        "--glob",
        default="Formated_Data/Experiment_1/**/video_*.csv",
        help="pattern for CSVs (relative or absolute); ** is recursive",
    )
    ap.add_argument(
        "--device",
        default=("mps" if torch.backends.mps.is_available()
                 else ("cuda" if torch.cuda.is_available() else "cpu")),
        help="device to run evaluation on",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=256,
        help="batch size for evaluation",
    )
    return ap.parse_args()


def split_list(xs, fracs=(0.7, 0.15, 0.15)):
    n = len(xs)
    a = int(fracs[0] * n)
    b = int((fracs[0] + fracs[1]) * n)
    return xs[:a], xs[a:b], xs[b:]


def stack_windows(seq_list, T, horizon, add_xyz_vel):
    Xs, Ys = [], []
    for df in seq_list:
        X, Y = make_windows(df, T=T, horizon=horizon, add_xyz_vel=add_xyz_vel)
        if len(X):
            Xs.append(X)
            Ys.append(Y)
    if not Xs:
        feat_dim = 8 + (3 if add_xyz_vel else 0)
        return (np.zeros((0, T, feat_dim), np.float32),
                np.zeros((0, 3), np.float32))
    return np.concatenate(Xs), np.concatenate(Ys)


@torch.no_grad()
def eval_metrics(model, dl, device="cpu"):
    model.to(device)
    model.eval()

    all_cos_losses = []
    all_angles_deg = []
    all_sqerr = []

    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        pred, _ = model(xb)  # (B,3)
        # cosine stuff (without pre-normalizing pred – same idea as cosine_loss)
        pn = pred.norm(dim=-1).clamp_min(EPS)
        tn = yb.norm(dim=-1).clamp_min(EPS)
        dot = (pred * yb).sum(dim=-1) / (pn * tn)
        dot = torch.clamp(dot, -1.0 + EPS, 1.0 - EPS)

        cos_loss = 1.0 - dot                # per-sample
        all_cos_losses.append(cos_loss.cpu())

        # angular error (deg)
        ang = torch.acos(dot) * (180.0 / math.pi)
        all_angles_deg.append(ang.cpu())

        # vector RMSE (difference in 3D space)
        diff = pred - yb
        sqerr = (diff * diff).sum(dim=-1)   # per-sample squared norm
        all_sqerr.append(sqerr.cpu())

    if not all_cos_losses:
        return {
            "mean_cosine_loss": float("nan"),
            "mean_angular_deg": float("nan"),
            "median_angular_deg": float("nan"),
            "rmse_vec": float("nan"),
            "hit5": float("nan"),
            "hit10": float("nan"),
        }

    cos_all = torch.cat(all_cos_losses)
    ang_all = torch.cat(all_angles_deg)
    sq_all = torch.cat(all_sqerr)

    mean_cos = cos_all.mean().item()
    mean_ang = ang_all.mean().item()
    median_ang = ang_all.median().item()
    rmse_vec = sq_all.mean().sqrt().item()

    hit5 = (ang_all <= 5.0).float().mean().item()
    hit10 = (ang_all <= 10.0).float().mean().item()

    return {
        "mean_cosine_loss": mean_cos,
        "mean_angular_deg": mean_ang,
        "median_angular_deg": median_ang,
        "rmse_vec": rmse_vec,
        "hit5": hit5,
        "hit10": hit10,
    }


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent

    # ----- find checkpoints -----
    ckpt_pattern = str(
        (root / args.ckpt_glob).resolve()
        if not Path(args.ckpt_glob).is_absolute()
        else Path(args.ckpt_glob)
    )
    ckpts = sorted(glob.glob(ckpt_pattern))
    if not ckpts:
        raise SystemExit(f"No checkpoints matched pattern: {ckpt_pattern}")
    print(f"[INFO] found {len(ckpts)} checkpoint(s):")
    for p in ckpts:
        print("  -", p)

    # ----- pick first TCN ckpt to get data config -----
    tcn_cfg = None
    for ckpt_path in ckpts:
        blob = torch.load(ckpt_path, map_location="cpu")
        cfg = blob.get("config", {}) or {}
        if cfg.get("arch") == "tcn":
            tcn_cfg = cfg
            print(f"\n[INFO] using data config from TCN checkpoint: {ckpt_path}")
            break

    if tcn_cfg is None:
        raise SystemExit("No TCN checkpoints found (config['arch'] == 'tcn').")

    hz = float(tcn_cfg.get("hz", 60.0))
    T = int(tcn_cfg.get("T", 20))
    horizon = int(tcn_cfg.get("horizon", 1))
    add_xyz_vel = bool(tcn_cfg.get("add_xyz_vel", 0))
    zscore = int(tcn_cfg.get("zscore", 0))

    print(f"  hz          = {hz}")
    print(f"  T           = {T}")
    print(f"  horizon     = {horizon}")
    print(f"  add_xyz_vel = {add_xyz_vel}")
    print(f"  zscore      = {bool(zscore)}")

    # ----- build dataset once -----
    pat_path = Path(args.glob)
    pattern = (root / pat_path).resolve() if not pat_path.is_absolute() else pat_path
    files = sorted(
        set(glob.glob(str(pattern), recursive=True))
        | set(glob.glob(str(pattern).replace(".csv", ".CSV"), recursive=True))
    )
    print(f"\n[INFO] matched CSV files: {len(files)}")
    if not files:
        raise SystemExit("No CSV files found; check --glob or folder structure.")

    need_len = T + horizon
    seqs = []
    for fp in files:
        df = load_and_resample(fp, hz=hz)
        if len(df) >= need_len:
            seqs.append(df)
    print(f"[INFO] usable sequences (>= {need_len} frames): {len(seqs)}")
    if not seqs:
        raise SystemExit("All sequences were too short after resampling.")

    train_seqs, val_seqs, test_seqs = split_list(seqs, (0.7, 0.15, 0.15))

    Xtr, Ytr = stack_windows(train_seqs, T, horizon, add_xyz_vel)
    Xva, Yva = stack_windows(val_seqs,   T, horizon, add_xyz_vel)
    Xte, Yte = stack_windows(test_seqs,  T, horizon, add_xyz_vel)
    print(f"[INFO] windows — train:{len(Xtr)}  val:{len(Xva)}  test:{len(Xte)}")
    if not len(Xte):
        raise SystemExit("No test windows; cannot compare models.")

    if zscore and len(Xtr) > 0:
        mu = Xtr.mean(axis=(0, 1), keepdims=True)
        sigma = Xtr.std(axis=(0, 1), keepdims=True).clip(1e-6)
        Xtr = (Xtr - mu) / sigma
        Xva = (Xva - mu) / sigma
        Xte = (Xte - mu) / sigma
        print("[INFO] applied z-score standardization to input features")

    in_dim = int(Xtr.shape[-1])
    print(f"[INFO] inferred in_dim = {in_dim}\n")

    dl_te = DataLoader(
        NpSeqDataset(Xte, Yte),
        batch_size=args.batch,
        shuffle=False,
    )

    # ----- evaluate each TCN checkpoint -----
    rows = []
    for ckpt_path in ckpts:
        blob = torch.load(ckpt_path, map_location="cpu")
        cfg = blob.get("config", {}) or {}
        arch = cfg.get("arch")

        if arch != "tcn":
            # we only compare TCN models here
            continue

        hidden = int(cfg.get("hidden", 96))
        layers = int(cfg.get("layers", 1))

        model = GazeTCN(in_dim=in_dim, hidden=hidden, layers=layers)
        model.load_state_dict(blob["model_state"])

        metrics = eval_metrics(model, dl_te, device=args.device)
        rows.append((Path(ckpt_path).name, metrics))

    if not rows:
        print("No TCN checkpoints to compare.")
        return

    # ----- pretty print table -----
    print("== Model comparison (each TCN evaluated on common test set) ==")
    header = (
        f"{'model':<20}"
        f"{'mean_cosine_loss':>18}"
        f"{'mean_angular_deg':>18}"
        f"{'median_angular_deg':>20}"
        f"{'rmse_vec':>12}"
        f"{'hit@5deg':>10}"
        f"{'hit@10deg':>10}"
    )
    print(header)
    print("-" * len(header))

    for name, m in rows:
        print(
            f"{name:<20}"
            f"{m['mean_cosine_loss']:>18.4f}"
            f"{m['mean_angular_deg']:>18.4f}"
            f"{m['median_angular_deg']:>20.4f}"
            f"{m['rmse_vec']:>12.4f}"
            f"{m['hit5']:>10.4f}"
            f"{m['hit10']:>10.4f}"
        )


if __name__ == "__main__":
    main()
