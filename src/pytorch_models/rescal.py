import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RESCAL(nn.Module):
    def __init__(self, out_features, dims, dropout=0, layernorm=False) -> None:
        super().__init__()
        self.dropout = dropout
        self.r_weights = nn.Parameter(Tensor(out_features, dims, dims))
        self.init_parameters()
        self.layernorm = None
        if layernorm:
            self.layernorm = nn.LayerNorm(out_features)

    def init_parameters(self) -> None:
        bound = 1 / math.sqrt(self.r_weights.size(1))
        nn.init.uniform_(self.r_weights, -bound, bound)

    def forward(self, head: Tensor, tail: Tensor) -> Tensor:
        # A -> (N x dims)
        # self.r_weights -> (nb_rels x dims x dims)

        out = torch.matmul(head, self.r_weights).matmul(tail.transpose(1, 0))
        # out -> (nb_rels x N x N)

        out = out.permute(1, 2, 0)
        # out -> (N x N x nb_rels)

        out = F.dropout(out, p=self.dropout, training=self.training)

        if self.layernorm is not None:
            out = self.layernorm(out)

        return out
