import torch
import torch.nn as nn
import torch.nn.functional as F

def safe_l2_normalize(x, dim=-1, eps=1e-8):
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))

class GazeLSTM(nn.Module):
    def __init__(self, in_dim=8, hidden=96, layers=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=in_dim, hidden_size=hidden,
            num_layers=layers, batch_first=True,
            dropout=(dropout if layers > 1 else 0.0)
        )
        self.head = nn.Linear(hidden, 3)

    def forward(self, x, h0=None):  # x: (B, T, in_dim)
        y, (h, c) = self.rnn(x, h0)  # y: (B, T, H)
        v = self.head(y[:, -1])      # (B, 3)
        return v, (h, c)
