import torch
import torch.nn as nn


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
            self.maxpools[i](self.relu(cov(x))).squeeze(3).squeeze(2)
            for i, cov in enumerate(self.conv)
        ]
        x = torch.cat(x, dim=1)  # B X Kn * len(Kz)
        y = self.fc(self.dropout(x))
        return y
