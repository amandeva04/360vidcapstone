import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from data_loader import HeadGazeDataset, get_filepaths_from_dir
from model import SequenceLSTM, SequenceGRU
from utils import angular_distance_rad

def collate_batch(batch):
    X = np.stack([item[0] for item in batch])
    y = np.stack([item[1] for item in batch])
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()

def load_model_from_checkpoint(ckpt_path, model_type='lstm', device='cpu', **model_kwargs):
    ckpt = torch.load(ckpt_path, map_location=device)
    if model_type.lower() == 'lstm':
        model = SequenceLSTM(**model_kwargs)
    else:
        model = SequenceGRU(**model_kwargs)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model

def evaluate_ckpt(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    filepaths = get_filepaths_from_dir(args.data_dir)
    _, _, test_files = [], [], filepaths  # if you want custom split, change here
    test_ds = HeadGazeDataset(test_files, seq_len=args.seq_len)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=lambda b: collate_batch(b))
    model = load_model_from_checkpoint(args.ckpt, model_type=args.model_type, device=device,
                                       input_size=7, hidden_size=args.hidden_size,
                                       num_layers=args.num_layers, bidirectional=args.bidirectional, dropout=args.dropout)
    angles = []
    with torch.no_grad():
        for X,y in test_loader:
            X = X.to(device); y = y.to(device)
            y_pred = model(X)
            ang = angular_distance_rad(y_pred, y)
            angles.append(ang.cpu().numpy())
    angles = np.concatenate(angles, axis=0)
    angles_deg = np.degrees(angles)
    print("Mean angle (deg):", angles_deg.mean())
    print("Median angle (deg):", np.median(angles_deg))
    print("Hit rate <=5°:", (angles_deg<=5.0).mean()*100)
    print("Hit rate <=10°:", (angles_deg<=10.0).mean()*100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="../data")
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='lstm')
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()
    evaluate_ckpt(args)
