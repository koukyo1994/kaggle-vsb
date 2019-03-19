import torch
import torch.nn as nn


class LSTMCNNNet(nn.Module):
    def __init__(self,
                 input_shape,
                 linear_size=100,
                 hidden_size=128,
                 o_channels=64):
        super(LSTMCNNNet, self).__init__()
        kernel_size = [3, 4, 5]
        self.maxlen = input_shape[1]
        self.input_dim = input_shape[2]
        self.lstm = nn.LSTM(
            self.input_dim, hidden_size, bidirectional=True, batch_first=True)
        self.conv = nn.ModuleList([
            nn.Conv2d(1, o_channels, (i, 2 * hidden_size)) for i in kernel_size
        ])
        self.maxpools = [
            nn.MaxPool2d((self.maxlen + 1 - i, 1)) for i in kernel_size
        ]
        self.relu = nn.ReLU()
        self.fc = nn.Linear(len(kernel_size) * o_channels, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        h_lstm, _ = self.lstm(x)
        h_lstm = h_lstm.unsqueeze(1)
        h_maxpool = [
            self.maxpools[i](self.relu(conv(h_lstm))).squeeze(3).squeeze(2)
            for i, conv in enumerate(self.conv)
        ]
        x = torch.cat(h_maxpool, dim=1)
        y = self.fc(self.dropout(x))
        return y
