import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, feature_dims, step_dims, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.support_masking = True

        self.bias = bias
        self.feature_dims = feature_dims
        self.step_dims = step_dims

        self.features_dim = 0

        weight = torch.zeros(feature_dims, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            b = torch.zeros(step_dims)
            nn.init.xavier_uniform_(b)
            self.b = nn.Parameter(b)

    def forward(self, x, mask=None):
        feature_dims = self.feature_dims
        step_dims = self.step_dims

        eij = torch.mm(x.contiguous().view(-1, feature_dims),
                       self.weight).view(-1, step_dims)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)
