import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_l2_normalize(x, dim=-1, eps=1e-8):
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))


class GazeLSTM(nn.Module):
    def __init__(self, in_dim=8, hidden=96, layers=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=(dropout if layers > 1 else 0.0),
        )
        self.head = nn.Linear(hidden, 3)

    def forward(self, x, h0=None):  # x: (B, T, in_dim)
        y, (h, c) = self.rnn(x, h0)  # y: (B, T, H)
        v = self.head(y[:, -1])      # (B, 3)
        return v, (h, c)


class CausalConv1d(nn.Module):
    """
    1D convolution that only looks at the past (causal).
    We pad on the left so output at time t never sees future inputs.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x):
        # x: (B, C, T)
        # F.pad pads as (left, right) along the time dimension
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """
    Residual TCN block:
      CausalConv → ReLU → Dropout → CausalConv → ReLU → Dropout → + residual
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.act   = nn.ReLU()
        self.drop  = nn.Dropout(dropout)

        # project residual if channels change
        if in_ch != out_ch:
            self.res_proj = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x):
        # x: (B, C, T)
        res = self.res_proj(x)
        h = self.conv1(x)
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.act(h)
        h = self.drop(h)
        return self.act(h + res)


class GazeTCN(nn.Module):
    """
    TCN-based gaze predictor.
    Input : (B, T, in_dim)
    Output: (B, 3) unnormalized 3D gaze vector (same as GazeLSTM head).
    """
    def __init__(self,
                 in_dim=8,
                 hidden=96,
                 layers=4,
                 kernel_size=3,
                 dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.layers = layers

        blocks = []
        in_ch = in_dim
        for i in range(layers):
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            blocks.append(
                TCNBlock(
                    in_ch=in_ch,
                    out_ch=hidden,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = hidden

        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, 3)

    def forward(self, x, h0=None):
        """
        x: (B, T, in_dim)
        We ignore h0 and hidden state, but keep the same signature as GazeLSTM.
        """
        # (B, T, C) -> (B, C, T) for Conv1d
        x = x.transpose(1, 2)
        h = self.tcn(x)          # (B, hidden, T)
        last = h[:, :, -1]       # last timestep: (B, hidden)
        v = self.head(last)      # (B, 3)
        # training/eval code only uses pred, ignores second return value
        return v, None
