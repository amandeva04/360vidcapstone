# main.py
import argparse, glob
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import load_and_resample, make_windows
from models import GazeLSTM, GazeTCN

from training import NpSeqDataset, fit, evaluate
from saving import save_checkpoint


# ---------------------------- CLI ----------------------------
def parse_args():
    ap = argparse.ArgumentParser("GazeGRU trainer")
    ap.add_argument("--hz", type=float, default=60.0, help="resample rate (Hz)")
    ap.add_argument("--T", type=int, default=20, help="window length")
    ap.add_argument("--horizon", type=int, default=1, help="steps ahead")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--device",
        default=("mps" if torch.backends.mps.is_available()
                 else ("cuda" if torch.cuda.is_available() else "cpu"))
    )
    ap.add_argument(
        "--glob",
        default="Formated_Data/Experiment_1/**/video_*.csv",
        help="pattern for CSVs (relative or absolute); ** is recursive",
    )
    # Feature toggles
    ap.add_argument("--add_xyz_vel", action="store_true", help="append x,y,z head-position velocities")
    ap.add_argument("--zscore", action="store_true", help="z-score standardize input features")

    # Architecture choice
    ap.add_argument(
        "--arch",
        choices=["lstm", "tcn"],
        default="lstm",
        help="model architecture (lstm or tcn)",
    )

    return ap.parse_args()


# ---------------------------- helpers ----------------------------
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
            Xs.append(X); Ys.append(Y)
    if not Xs:
        feat_dim = 8 + (3 if add_xyz_vel else 0)  # cos/sin yaw/pitch + d/dt + d2/dt2 [+ xyz vel]
        return (np.zeros((0, T, feat_dim), np.float32),
                np.zeros((0, 3), np.float32))
    return np.concatenate(Xs), np.concatenate(Ys)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------- main ----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    ROOT = Path(__file__).resolve().parent

    # Resolve glob pattern robustly (works regardless of cwd)
    pat_path = Path(args.glob)
    pattern = str((ROOT / pat_path).resolve()) if not pat_path.is_absolute() else str(pat_path)

    # Match lowercase + uppercase
    files = sorted(set(glob.glob(pattern, recursive=True)) |
                   set(glob.glob(pattern.replace(".csv", ".CSV"), recursive=True)))
    print(f"[INFO] script dir = {ROOT}")
    print(f"[INFO] pattern    = {pattern}")
    print(f"[INFO] matched files: {len(files)}")
    for p in files[:8]:
        print("  -", p)
    assert files, "No CSV files found; check --glob or folder structure."

    # 1) Load & resample per file
    seqs = []
    need_len = args.T + args.horizon
    for fp in files:
        df = load_and_resample(fp, hz=args.hz)
        if len(df) >= need_len:
            seqs.append(df)
    print(f"[INFO] usable sequences (>= {need_len} frames): {len(seqs)}")
    assert seqs, "All sequences were too short after resampling."

    # 2) Split by videos (no leakage)
    train_seqs, val_seqs, test_seqs = split_list(seqs, (0.7, 0.15, 0.15))

    # 3) Window & stack with selected feature set
    Xtr, Ytr = stack_windows(train_seqs, args.T, args.horizon, args.add_xyz_vel)
    Xva, Yva = stack_windows(val_seqs,   args.T, args.horizon, args.add_xyz_vel)
    Xte, Yte = stack_windows(test_seqs,  args.T, args.horizon, args.add_xyz_vel)
    print(f"[INFO] windows — train:{len(Xtr)}  val:{len(Xva)}  test:{len(Xte)}")
    assert len(Xtr) and len(Xva), "Need non-empty train/val windows."

    # 3b) Optional standardization (recommended for vel/acc channels)
    if args.zscore and len(Xtr) > 0:
        mu = Xtr.mean(axis=(0,1), keepdims=True)
        sigma = Xtr.std(axis=(0,1), keepdims=True).clip(1e-6)
        Xtr = (Xtr - mu) / sigma
        Xva = (Xva - mu) / sigma
        Xte = (Xte - mu) / sigma
        print("[INFO] applied z-score standardization to input features")

    # 4) DataLoaders
    dl_tr = DataLoader(NpSeqDataset(Xtr, Ytr), batch_size=args.batch, shuffle=True)
    dl_va = DataLoader(NpSeqDataset(Xva, Yva), batch_size=args.batch, shuffle=False)
    dl_te = DataLoader(NpSeqDataset(Xte, Yte), batch_size=args.batch, shuffle=False)

    # 5) TensorBoard writer
    arch_tag = args.arch.upper()
    run_name = (
        f"Gaze{arch_tag}_T{args.T}_H{args.hidden}_L{args.layers}_"
        f"hz{int(args.hz)}_{'xyzv' if args.add_xyz_vel else 'noxyzv'}_"
        f"{'z' if args.zscore else 'raw'}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    logdir = ROOT / "runs" / run_name
    writer = SummaryWriter(str(logdir))
    print(f"[TB] logging to {logdir}")

    # 6) Model (auto-infer in_dim)
    in_dim = int(Xtr.shape[-1])
    print(f"[INFO] inferred in_dim = {in_dim}")

    if args.arch == "lstm":
        print("[INFO] using GazeLSTM")
        model = GazeLSTM(in_dim=in_dim, hidden=args.hidden, layers=args.layers).to(args.device)
    elif args.arch == "tcn":
        print("[INFO] using GazeTCN")
        model = GazeTCN(in_dim=in_dim, hidden=args.hidden, layers=args.layers).to(args.device)
    else:
        raise ValueError(f"Unknown arch: {args.arch}")

    # Train
    best = fit(
        model,
        dl_tr,
        dl_va,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        clip=None,                 # <— let grads breathe at first
        writer=writer,
    )
    # always have a usable state_dict
    if not isinstance(best.get("state_dict"), dict):
        print("[WARN] No valid best state captured; using current model weights.")
        best["state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        best["val_deg"] = float("inf")

    model.load_state_dict(best["state_dict"])

    # 7) Test
    test_deg = evaluate(model, dl_te, device=args.device)
    print(f"[RESULT] Test Angular Error: {test_deg:.2f}°  (best val: {best['val_deg']:.2f}°)")

    # (optional) hparams summary in TensorBoard
    hparams = {
        "T": args.T,
        "horizon": args.horizon,
        "hz": args.hz,
        "hidden": args.hidden,
        "layers": args.layers,
        "batch": args.batch,
        "lr": args.lr,
        "epochs": args.epochs,
        "device": args.device,
        "in_dim": in_dim,
        "add_xyz_vel": int(args.add_xyz_vel),
        "zscore": int(args.zscore),
        "arch": args.arch,
    }
    writer.add_hparams(
        hparams,
        {
            "hparams/best_val_deg": float(best["val_deg"]),
            "hparams/test_deg": float(test_deg),
        },
    )
    writer.close()

    # 8) Save into Trained_Models/
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "Trained_Models"
    out_name = (
        f"Gaze{arch_tag}_T{args.T}_H{args.hidden}_L{args.layers}_"
        f"hz{int(args.hz)}_ep{args.epochs}_in{in_dim}_{stamp}.pt"
    )
    out_path = out_dir / out_name
    cfg = vars(args) | {
        "test_deg": float(test_deg),
        "best_val_deg": float(best["val_deg"]),
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "n_test": int(len(Xte)),
        "in_dim": in_dim,
        "run_dir": str(logdir),
    }
    save_checkpoint(str(out_path), best["state_dict"], cfg)
    print(f"[SAVE] {out_path}")
    save_checkpoint(str(out_dir / "latest.pt"), best["state_dict"], cfg)
    print(f"[SAVE] {out_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
