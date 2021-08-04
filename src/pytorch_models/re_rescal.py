from src.pytorch_models.mlp import MLP
from src.pytorch_models.rescal import RESCAL
from torch import nn


class ReRescal(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dropout: float = 0.05,
        act: str = 'relu',
        layernorm: bool = False,
    ):
        super().__init__()
        self.re = RESCAL(
            out_features=out_features,
            dims=200,
            dropout=dropout,
            layernorm=layernorm
        )
        self.out_features = out_features

        self.head_mlp = MLP(
            input_features=in_features,
            arch=[],
            output_features=200,
            dropout=dropout,
            layernorm=layernorm,
            act=act,
        )

        self.tail_mlp = MLP(
            input_features=in_features,
            arch=[],
            output_features=200,
            dropout=dropout,
            layernorm=layernorm,
            act=act,
        )

    def forward(self, x):
        head = self.head_mlp(x)
        tail = self.tail_mlp(x)
        return self.re(head, tail)