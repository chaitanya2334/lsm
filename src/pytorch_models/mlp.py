from typing import List, Union

from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        arch: List[int] = list(),
        act: Union[str, bool] = 'tanh',
        dropout: Union[float, bool] = False,
        layernorm: bool = False,
        bias: bool = True
    ):
        super().__init__()

        arch = [input_features] + list(arch) + [output_features]
        layers = []
        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i + 1], bias=bias))
            if act:
                layers.append(self.activation(act))
            if dropout:
                layers.append(nn.Dropout(p=dropout))

        self.mlp = nn.Sequential(*layers)
        self.layernorm = None
        if layernorm:
            self.layernorm = nn.LayerNorm(output_features)

    def activation(self, act_type):
        if act_type == "tanh":
            return nn.Tanh()
        elif act_type == "relu":
            return nn.ReLU()
        elif act_type == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"{act_type} not supported")

    def forward(self, x):
        x = self.mlp(x)
        if self.layernorm is not None:
            x = self.layernorm(x)

        return x
