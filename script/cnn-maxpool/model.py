import torch.nn as nn


class CNNMaxPool(nn.Module):
    def __init__(self, o_channels, kernels=[3, 4, 5]):
        super(CNNMaxPool, self).__init__()

        self.o_channels = o_channels
        self.kernels = kernels

        self.cnn = [nn.Conv1d(1, o_channels, k) for k in kernels]
