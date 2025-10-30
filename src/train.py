import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from .model import SequenceLSTM, SequenceGRU
from .utils import angular_loss, angular_distance_rad

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_filepaths_from_dir(data_dir):
    # Recursively get CSV files
    filepaths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".csv") and 'videoMeta' not in f and 'userDemo' not in f:
                filepaths.append(os.path.join(root, f))
    return filepaths

def split_filepaths(filepaths, val_ratio=0.2, test_ratio=0.1):
    random.shuffle(filepaths)
    N = len(filepaths)
    n_test = max(1, int(N * test_ratio))
    n_val = max(1, int(N * val_ratio))
    test = filepaths[:n_test]
    val = filepaths[n_test:n_test+n_val]
    train = filepaths[n_test+n_val:]
    return train, val, test

class HeadGazeDataset(torch.utils.data.Dataset):
    def __init__(self, filepaths, seq_len=30):
        self.sequences = []
        self.seq_len = seq_len
        for fp in filepaths:
            data = np.loadtxt(fp, delimiter=',', skiprows=1)
            # skip if too short
            if data.shape[0] < seq_len:
                continue
            # create sliding windows
            for i in range(len(data) - seq_len):
                X = data[i:i+seq_len, :]  # all 7 columns as input
                y = data[i+seq_len, :]    # all 7 columns as target
                self.sequences.append((X, y))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X, y = self.sequences[idx]
        return X, y

def collate_batch(batch):
    X = np.stack([item[0] for item in batch])
    y = np.stack([item[1][4:7] for item in batch])  # take only x,y,z for target
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()

def train_loop(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    filepaths = get_filepaths_from_dir(args.data_dir)
    train_files, val_files, test_files = split_filepaths(filepaths, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    train_ds = HeadGazeDataset(train_files, seq_len=args.seq_len)
    val_ds = HeadGazeDataset(val_files, seq_len=args.seq_len)
    test_ds = HeadGazeDataset(test_files, seq_len=args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    # model
    if args.model_type.lower() == 'lstm':
        model = SequenceLSTM(input_size=7, hidden_size=args.hidden_size, num_layers=args.num_layers,
                             bidirectional=args.bidirectional, dropout=args.dropout)
    else:
        model = SequenceGRU(input_size=7, hidden_size=args.hidden_size, num_layers=args.num_layers,
                            bidirectional=args.bidirectional, dropout=args.dropout)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    train_losses, val_losses = [], []
    best_val = 1e9

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} train")
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = angular_loss(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            pbar.set_postfix({'batch_loss': loss.item()})

        epoch_loss = running_loss / len(train_ds)
        train_losses.append(epoch_loss)

        # validation
        model.eval()
        v_running = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_pred = model(X_val)
                v_loss = angular_loss(y_pred, y_val)
                v_running += v_loss.item() * X_val.size(0)
            val_epoch_loss = v_running / len(val_ds)
            val_losses.append(val_epoch_loss)

        scheduler.step(val_epoch_loss)
        print(f"Epoch {epoch}: train_loss={epoch_loss:.6f}, val_loss={val_epoch_loss:.6f}")

        # save best model
        if val_epoch_loss < best_val:
            best_val = val_epoch_loss
            ckpt_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({'model_state': model.state_dict(),
                        'args': vars(args)}, ckpt_path)
            print(f"Saved best model to {ckpt_path}")

    # save logs and plots
    np.save(os.path.join(args.output_dir, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(args.output_dir, 'val_losses.npy'), np.array(val_losses))
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Angular Loss (rad)')
    plt.legend()
    plt.title('Training & Validation Loss')
    plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'))
    plt.close()

    # evaluate test
    evaluate(model, test_loader, device, args)

def evaluate(model, loader, device, args):
    model.eval()
    angles = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            ang = angular_distance_rad(y_pred, y)
            angles.append(ang.cpu().numpy())
    angles = np.concatenate(angles, axis=0)
    angles_deg = np.degrees(angles)
    mean_ang, med_ang = angles_deg.mean(), np.median(angles_deg)
    hit_rate_5 = (angles_deg <= 5.0).mean() * 100
    hit_rate_10 = (angles_deg <= 10.0).mean() * 100

    print(f"Test results: mean_ang={mean_ang:.3f} deg, median={med_ang:.3f} deg")
    print(f"Hit rate <=5°: {hit_rate_5:.2f} %, <=10°: {hit_rate_10:.2f} %")

    plt.figure()
    plt.hist(angles_deg, bins=50)
    plt.xlabel('Angular error (deg)')
    plt.ylabel('Count')
    plt.title('Histogram of angular errors (test)')
    plt.savefig(os.path.join(args.output_dir, 'angles_hist.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="Formated_Data")
    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm','gru'])
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()
    train_loop(args)
