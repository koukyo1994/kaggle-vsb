import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, feature_dims, step_dims, n_middle, n_attention,
                 **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.support_masking = True
        self.feature_dims = feature_dims
        self.step_dims = step_dims
        self.n_middle = n_middle
        self.n_attention = n_attention
        self.features_dim = 0

        self.lin1 = nn.Linear(feature_dims, n_middle, bias=False)
        self.lin2 = nn.Linear(n_middle, n_attention, bias=False)

    def forward(self, x, mask=None):
        step_dims = self.step_dims

        eij = self.lin1(x)
        eij = torch.tanh(eij)
        eij = self.lin2(eij)

        a = torch.exp(eij).reshape(-1, self.n_attention, step_dims)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 2, keepdim=True) + 1e-10

        weighted_input = torch.bmm(a, x)
        return torch.sum(weighted_input, 1)


class LSTMGRUAttentionNet(nn.Module):
    def __init__(self, hidden_size, linear_size, input_shape, n_attention):
        super(LSTMGRUAttentionNet, self).__init__()

        self.maxlen = input_shape[1]
        self.input_dim = input_shape[2]
        self.lstm = nn.LSTM(
            self.input_dim, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(
            hidden_size * 2,
            int(hidden_size / 2),
            bidirectional=True,
            batch_first=True)
        self.attn = Attention(
            int(hidden_size / 2) * 2, self.maxlen, n_attention, n_attention)
        self.lin1 = nn.Linear(int(hidden_size / 2) * 2, linear_size)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(linear_size, 1)
