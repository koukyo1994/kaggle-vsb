import torch
import torch.nn as nn

from script.common.layers import Attention


class LSTMAttentionNet(nn.Module):
    def __init__(self, hidden_size, linear_size, input_shape, n_attention):
        super(LSTMAttentionNet, self).__init__()

        self.maxlen = input_shape[1]
        self.input_dim = input_shape[2]
        self.lstm1 = nn.LSTM(
            self.input_dim, hidden_size, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(
            hidden_size * 2,
            int(hidden_size / 2),
            bidirectional=True,
            batch_first=True)
        self.attn = Attention(
            int(hidden_size / 2) * 2, self.maxlen, n_attention, n_attention)
        self.lin1 = nn.Linear(int(hidden_size / 2) * 2, linear_size)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(linear_size, 1)

    def forward(self, x):
        h_lstm1, _ = self.lstm1(x)
        h_lstm2, _ = self.lstm2(h_lstm1)
        attn = self.attn(h_lstm2)
        lin = self.relu(self.lin1(attn))
        out = self.lin2(lin)
        return out


class CNNBaseModel(nn.Module):
    def __init__(self, linear_size=100, o_channels=64, input_shape=()):
        super(CNNBaseModel, self).__init__()
        kernel_size = [3, 4, 5]

        self.conv = nn.ModuleList([
            nn.Conv2d(1, o_channels, (i, input_shape[2])) for i in kernel_size
        ])
        self.maxpools = [
            nn.MaxPool2d((input_shape[1] + 1 - i, 1)) for i in kernel_size
        ]
        self.fc = nn.Linear(len(kernel_size) * o_channels, 1)
        self.dropout = nn.Dropout(0.2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = [
            self.maxpools[i](self.relu(cov(
                x.unsqueeze(1)))).squeeze(3).squeeze(2)
            for i, cov in enumerate(self.conv)
        ]
        x = torch.cat(x, dim=1)  # B X Kn * len(Kz)
        y = self.fc(self.dropout(x))
        return y
