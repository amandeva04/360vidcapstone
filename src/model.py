import torch
import torch.nn as nn

class SequenceLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=2, bidirectional=False, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers>1 else 0.0,
                            bidirectional=bidirectional)
        out_dim = hidden_size * (2 if bidirectional else 1)
        # predict a 3D gaze vector
        self.fc = nn.Sequential(
            nn.Linear(out_dim, out_dim//2),
            nn.ReLU(),
            nn.Linear(out_dim//2, 3)
        )

    def forward(self, x):
        # x: (B, T, input_size)
        out, (hn, cn) = self.lstm(x)  # out: (B, T, hidden*directions)
        # take last timestep
        last = out[:, -1, :]
        y = self.fc(last)  # (B,3)
        return y


class SequenceGRU(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=2, bidirectional=False, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers>1 else 0.0,
                          bidirectional=bidirectional)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(out_dim, out_dim//2),
            nn.ReLU(),
            nn.Linear(out_dim//2, 3)
        )

    def forward(self, x):
        out, hn = self.gru(x)
        last = out[:, -1, :]
        y = self.fc(last)
        return y
