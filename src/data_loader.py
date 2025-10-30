import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

def get_filepaths_from_dir(data_dir):
    """
    Recursively find all .csv files under data_dir.
    Returns a list of filepaths.
    """
    csv_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, f))
    print(f"Found {len(csv_files)} CSV files in {data_dir}")
    return csv_files


class HeadGazeDataset(torch.utils.data.Dataset):
    def __init__(self, filepaths, seq_len=30):
        self.sequences = []
        self.seq_len = seq_len
        for fp in filepaths:
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                print(f"Skipping {fp}, could not read: {e}")
                continue

            # keep only the numeric columns we care about
            required = ['qw','qx','qy','qz','x','y','z']
            if not all(col in df.columns for col in required):
                continue

            data = df[required].values.astype(np.float32)

            if data.shape[0] < seq_len + 1:
                continue

            # sliding window
            for i in range(len(data) - seq_len):
                X = data[i:i+seq_len, :]
                y = data[i+seq_len, 4:7]  # x,y,z only
                self.sequences.append((X, y))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X, y = self.sequences[idx]
        return X, y
