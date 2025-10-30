import torch
import numpy as np
import matplotlib.pyplot as plt
from model import SequenceLSTM
from data_loader import HeadGazeDataset, get_filepaths_from_dir
from utils import angular_distance_rad

def sample_predictions(model, dataset, device, n=50):
    model.eval()
    inds = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    preds = []
    trues = []
    with torch.no_grad():
        for i in inds:
            X,y = dataset[i]
            X = torch.from_numpy(X[None]).float().to(device)
            y_pred = model(X).cpu().numpy().squeeze()
            preds.append(y_pred)
            trues.append(y)
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    angs = np.degrees(angular_distance_rad(torch.from_numpy(preds), torch.from_numpy(trues)).numpy())
    # scatter predicted vs true angle (dot product)
    plt.figure()
    plt.scatter(angs, np.zeros_like(angs), alpha=0.6)
    plt.xlabel("Angular error (deg)")
    plt.title("Sample prediction angular errors")
    plt.savefig("outputs/sample_pred_angles.png")
    plt.close()

if __name__ == "__main__":
    # quick demo usage: change as needed
    pass
