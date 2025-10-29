# training.py
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import trange, tqdm

from lossfunction import angular_loss, angular_error_deg

class NpSeqDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert isinstance(X, np.ndarray) and isinstance(Y, np.ndarray)
        assert len(X) == len(Y)
        self.X, self.Y = X, Y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])

def fit(model,
        dl_tr: DataLoader,
        dl_va: DataLoader,
        epochs: int = 12,
        lr: float = 1e-3,
        device: str = "cpu",
        clip: float | None = 1.0,
        show_batches: bool = True,
        writer=None,                 # <-- TensorBoard SummaryWriter (optional)
        log_batches: bool = False    # <-- set True to also log per-batch
        ):
    """
    Train model; return {"val_deg": float, "state_dict": OrderedDict}.
    If `writer` is provided, logs per-epoch (and optional per-batch) metrics to TensorBoard.
    """
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # ensure 'best' is always usable
    best = {
        "val_deg": float("inf"),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}
    }

    global_step = 0  # for per-batch logging, if enabled
    epoch_bar = trange(1, epochs + 1, desc="Epochs", unit="ep")

    for ep in epoch_bar:
        # ---- Train ----
        model.train()
        tr_losses, tr_degs = [], []
        it = dl_tr if not show_batches else tqdm(dl_tr, leave=False, desc=f"train {ep}/{epochs}")
        for xb, yb in it:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred, _ = model(xb)
            loss = angular_loss(pred, yb)         # radians
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

            # metrics
            tr_losses.append(loss.item())
            with torch.no_grad():
                tr_degs.append(angular_error_deg(pred, yb))

            # per-batch TB logs (optional)
            if writer is not None and log_batches:
                writer.add_scalar("batch/train_loss_rad", loss.item(), global_step)
                writer.add_scalar("batch/train_deg", tr_degs[-1], global_step)

            global_step += 1
            if show_batches:
                it.set_postfix(loss=f"{np.mean(tr_losses):.4f}",
                               deg=f"{np.mean(tr_degs):.2f}°")

        # ---- Validate ----
        model.eval()
        va_degs, va_losses = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                pred, _ = model(xb)
                va_degs.append(angular_error_deg(pred, yb))
                va_losses.append(angular_loss(pred, yb).item())
        val_deg  = float(np.mean(va_degs))  if va_degs  else float("nan")
        val_loss = float(np.mean(va_losses)) if va_losses else float("nan")

        # epoch progress bar
        if math.isnan(val_deg) or math.isinf(val_deg):
            epoch_bar.set_postfix(val="nan", train=f"{np.mean(tr_degs):.2f}°")
        else:
            epoch_bar.set_postfix(val=f"{val_deg:.2f}°", train=f"{np.mean(tr_degs):.2f}°")

        # ---- TensorBoard epoch logs ----
        if writer is not None:
            writer.add_scalar("epoch/train_loss_rad", float(np.mean(tr_losses)), ep)
            writer.add_scalar("epoch/train_deg",      float(np.mean(tr_degs)),   ep)
            writer.add_scalar("epoch/val_loss_rad",   val_loss,                  ep)
            writer.add_scalar("epoch/val_deg",        val_deg,                   ep)
            # (optional) learning rate
            if hasattr(opt, "param_groups") and len(opt.param_groups):
                writer.add_scalar("epoch/lr", opt.param_groups[0]["lr"], ep)

        # ---- Track best ----
        if not (math.isnan(val_deg) or math.isinf(val_deg)):
            if val_deg < best["val_deg"]:
                best["val_deg"] = val_deg
                best["state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    return best

@torch.no_grad()
def evaluate(model, dl: DataLoader, device: str = "cpu"):
    model.to(device)
    model.eval()
    vals = []
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        pred, _ = model(xb)
        vals.append(angular_error_deg(pred, yb))
    return float(np.mean(vals)) if vals else float("nan")
