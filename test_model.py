import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from saving import load_checkpoint
from models import GazeGRU
from data import load_and_resample, make_windows
from lossfunction import angular_error_deg

def resolve_csv_paths(path_str: str):
    p = Path(path_str)

    if p.is_dir():
        return sorted(str(x) for x in p.rglob("*.csv"))
    if p.is_file() and p.suffix.lower() == ".csv":
        return [str(p)]
    return sorted(glob.glob(path_str, recursive=True))

def load_test_windows(test_path: str, T: int, horizon: int):
    files = resolve_csv_paths(test_path)
    if not files:
        raise ValueError(f"No CSV files found under: {test_path}")

    print(f"[INFO] Found {len(files)} test CSV files.")

    Xs, Ys = [], []
    need_len = T + horizon

    for fp in files:
        try:
            df = load_and_resample(fp, kalman_filter=True)
        except Exception as e:
            print(f"[WARN] Failed to load {fp}: {e}")
            continue

        if len(df) < need_len:
            continue

        X, Y = make_windows(df, T=T, horizon=horizon)
        if len(X):
            Xs.append(X)
            Ys.append(Y)

    if not Xs:
        raise ValueError("No valid windows extracted from test dataset.")

    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)

    print(f"[INFO] Test windows: X={X.shape}, Y={Y.shape}")

    return X, Y

def evaluate_model(model: GazeGRU, X: torch.Tensor, Y: torch.Tensor):
    model.eval()
    with torch.no_grad():
        preds, _ = model(X)

        pred_norm = preds / preds.norm(dim=1, keepdim=True).clamp_min(1e-8)
        true_norm = Y     / Y.norm(dim=1, keepdim=True).clamp_min(1e-8)

        dots = (pred_norm * true_norm).sum(dim=1).clamp(-1, 1)
        ang_all = torch.acos(dots) * (180.0 / np.pi)

        #ang_mean = ang_all.mean().item()
        lf_mean = angular_error_deg(pred_norm, true_norm)

        hr_5  = (ang_all <= 5).float().mean().item()
        hr_10 = (ang_all <= 10).float().mean().item()
        hr_15 = (ang_all <= 15).float().mean().item()

    return {
        "angular_error_deg": lf_mean,
        "hit_rate_5deg": hr_5,
        "hit_rate_10deg": hr_10,
        "hit_rate_15deg": hr_15,
    }

def test_model(model_path: str, test_path: str):
    print(f"[INFO] Loading model checkpoint: {model_path}")

    dummy = GazeGRU()
    config = load_checkpoint(model_path, dummy)

    model = GazeGRU(
        in_dim=config.get("in_dim", 8),
        hidden=config.get("hidden", 96),
        layers=config.get("layers", 1),
        dropout=config.get("dropout", 0.0)
    )
    load_checkpoint(model_path, model)

    print("[INFO] Loaded model with:")
    print(f"       in_dim={model.rnn.input_size}")
    print(f"       hidden={model.rnn.hidden_size}")
    print(f"       layers={model.rnn.num_layers}")

    T = config.get("T", 20)
    horizon = config.get("horizon", 1)

    print(f"[INFO] Using window params: T={T}, horizon={horizon}")
    X_np, Y_np = load_test_windows(test_path, T, horizon)

    X = torch.from_numpy(X_np)
    Y = torch.from_numpy(Y_np)

    results = evaluate_model(model, X, Y)

    print("\n========== TEST RESULTS ==========")
    print(f"Mean Angular Error: {results['angular_error_deg']:.3f}°")
    print(f"Hit-Rate @ 5°  : {results['hit_rate_5deg'] * 100:.2f}%")
    print(f"Hit-Rate @ 10° : {results['hit_rate_10deg'] * 100:.2f}%")
    print(f"Hit-Rate @ 15° : {results['hit_rate_15deg'] * 100:.2f}%")
    print("=================================\n")

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained gaze model (.pt).")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to .pt checkpoint of trained model."
    )

    parser.add_argument(
        "--glob",
        type=str,
        default="Formated_Data/Experiment_2/**/video_*.csv",
        help="Path, folder, or glob for test CSV files."
    )

    args = parser.parse_args()
    print(f"Dataset filepath: {args.glob}")
    test_model(args.model, args.glob)


if __name__ == "__main__":
    main()