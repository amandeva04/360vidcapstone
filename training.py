# training.py
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import trange, tqdm

# ⬇️ swap imports: use cosine_loss for training, keep angular_error_deg for logging
from lossfunction import cosine_loss, angular_error_deg

class NpSeqDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert isinstance(X, np.ndarray) and isinstance(Y, np.ndarray)
        assert len(X) == len(Y)
        self.X, self.Y = X, Y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        # ensure float32 tensors (your arrays are already float32, but this keeps it explicit)
        return torch.as_tensor(self.X[i], dtype=torch.float32), torch.as_tensor(self.Y[i], dtype=torch.float32)

def fit(model,
        dl_tr: DataLoader,
        dl_va: DataLoader,
        epochs: int = 12,
        lr: float = 1e-3,
        device: str = "cpu",
        clip: float | None = None,           # ⬅ default None (let grads breathe first)
        show_batches: bool = True,
        writer=None,
        log_batches: bool = False,
        ):
    """
    Train model; return {"val_deg": float, "state_dict": OrderedDict}.
    Logs per-epoch metrics to TensorBoard if `writer` is provided.
    """
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    best = {
        "val_deg": float("inf"),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}
    }

    global_step = 0
    epoch_bar = trange(1, epochs + 1, desc="Epochs", unit="ep")

    for ep in epoch_bar:
        # ---- Train ----
        model.train()
        tr_losses, tr_degs = [], []
        it = dl_tr if not show_batches else tqdm(dl_tr, leave=False, desc=f"train {ep}/{epochs}")
        for xb, yb in it:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred, _ = model(xb)

            # ⬇️ smoother training objective
            loss = cosine_loss(pred, yb)

            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            # simple NaN/Inf guard
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {global_step}: {loss.item()}")

            opt.step()

            tr_losses.append(loss.item())
            with torch.no_grad():
                tr_degs.append(angular_error_deg(pred, yb))

            if writer is not None and log_batches:
                writer.add_scalar("batch/train_cosine_loss", loss.item(), global_step)
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
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                pred, _ = model(xb)
                va_degs.append(angular_error_deg(pred, yb))
                # report cosine loss on val too (consistent with train)
                va_losses.append(cosine_loss(pred, yb).item())
        val_deg  = float(np.mean(va_degs))  if va_degs  else float("nan")
        val_loss = float(np.mean(va_losses)) if va_losses else float("nan")

        if math.isnan(val_deg) or math.isinf(val_deg):
            epoch_bar.set_postfix(val="nan", train=f"{np.mean(tr_degs):.2f}°")
        else:
            epoch_bar.set_postfix(val=f"{val_deg:.2f}°", train=f"{np.mean(tr_degs):.2f}°")

        # ---- TensorBoard epoch logs ----
        if writer is not None:
            writer.add_scalar("epoch/train_cosine_loss", float(np.mean(tr_losses)), ep)
            writer.add_scalar("epoch/train_deg",          float(np.mean(tr_degs)),   ep)
            writer.add_scalar("epoch/val_cosine_loss",    val_loss,                  ep)
            writer.add_scalar("epoch/val_deg",            val_deg,                   ep)
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
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        pred, _ = model(xb)
        vals.append(angular_error_deg(pred, yb))
    return float(np.mean(vals)) if vals else float("nan")
