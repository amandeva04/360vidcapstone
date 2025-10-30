import numpy as np
import torch
import matplotlib.pyplot as plt

def normalize_features(arr):
    """Normalize each column to zero mean, unit variance. Returns normalized arr."""
    mean = np.mean(arr, axis=0, keepdims=True)
    std = np.std(arr, axis=0, keepdims=True) + 1e-9
    return (arr - mean) / std

def quat_to_rotation_matrix(q):
    """
    q: (N,4) with [qw,qx,qy,qz]
    returns R: (N,3,3)
    """
    qw = q[:,0]; qx = q[:,1]; qy = q[:,2]; qz = q[:,3]
    # compute terms
    R = np.zeros((q.shape[0], 3, 3), dtype=np.float32)
    R[:,0,0] = 1 - 2*(qy*qy + qz*qz)
    R[:,0,1] = 2*(qx*qy - qz*qw)
    R[:,0,2] = 2*(qx*qz + qy*qw)
    R[:,1,0] = 2*(qx*qy + qz*qw)
    R[:,1,1] = 1 - 2*(qx*qx + qz*qz)
    R[:,1,2] = 2*(qy*qz - qx*qw)
    R[:,2,0] = 2*(qx*qz - qy*qw)
    R[:,2,1] = 2*(qy*qz + qx*qw)
    R[:,2,2] = 1 - 2*(qx*qx + qy*qy)
    return R

def quat_to_forward_vector(quats):
    """
    Convert unit quaternion to forward vector by rotating local forward (0,0,1).
    quats: (N,4) [qw,qx,qy,qz]
    returns (N,3) unit vectors
    """
    R = quat_to_rotation_matrix(quats)  # (N,3,3)
    local_forward = np.array([0., 0., 1.], dtype=np.float32)
    fwd = R @ local_forward  # broadcasting -> (N,3)
    # ensure normalization
    norms = np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-12
    return (fwd / norms).astype(np.float32)


### Angular loss & metrics (PyTorch tensors)
def angular_distance_rad(pred, target, eps=1e-7):
    """
    pred, target: tensors shape (batch,3) or (...,3), assumed not necessarily unit-length.
    returns: angle in radians
    """
    # ensure float
    pred = pred.float()
    target = target.float()
    # normalize
    pred_n = pred / (pred.norm(dim=-1, keepdim=True) + eps)
    targ_n = target / (target.norm(dim=-1, keepdim=True) + eps)
    cos = (pred_n * targ_n).sum(dim=-1).clamp(-1.0, 1.0)
    ang = torch.acos(cos)
    return ang

def angular_loss(pred, target):
    """
    Angular loss = mean(angle in radians)
    """
    return angular_distance_rad(pred, target).mean()


def plot_loss_curve(train_losses, val_losses=None, save_path=None):
    """
    Plot training (and optional validation) loss curve.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    if val_losses is not None:
        plt.plot(val_losses, label='Val Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_angular_error_histogram(errors, save_path=None):
    """
    Plot histogram of angular errors (in degrees).
    """
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=30, color='green', alpha=0.7)
    plt.xlabel('Angular Error (degrees)')
    plt.ylabel('Frequency')
    plt.title('Angular Error Distribution')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()