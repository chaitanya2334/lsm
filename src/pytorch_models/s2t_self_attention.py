from typing import List

import torch
from src.pytorch_models.mlp import MLP
from src.utils import min_value_of_dtype, tiny_value_of_dtype
from torch import nn


class SelfAttentionS2T(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        value_out_dim: int,
        use_value_mlp: bool = False,
        query_mlp_arch: List[int] = None,
        value_mlp_arch: List[int] = None
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_value_mlp = use_value_mlp
        self.out_dim = value_out_dim

        self.query_mlp = MLP(
            input_features=emb_dim,
            output_features=1,
            act=False,
            arch=query_mlp_arch if query_mlp_arch else [],
            bias=True
        )

        self.value_mlp = MLP(
            input_features=emb_dim,
            output_features=value_out_dim,
            act=False,
            arch=value_mlp_arch if value_mlp_arch else [],
            bias=True
        )

    def get_output_dim(self):
        return self.out_dim

    def masked_softmax(
        self,
        vector: torch.Tensor,
        mask: torch.BoolTensor,
        dim: int = -1,
        memory_efficient: bool = False,
    ) -> torch.Tensor:
        """
        from https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py#L247
        """
        if mask is None:
            result = nn.functional.softmax(vector, dim=dim)

        else:
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)

            if not memory_efficient:
                # To limit numerical errors from large vector elements outside
                # the mask, we zero these out.
                result = nn.functional.softmax(vector * mask, dim=dim)
                result = result * mask
                result = result / (
                    result.sum(dim=dim, keepdim=True)
                    + tiny_value_of_dtype(result.dtype)
                )

            else:
                masked_vector = vector.masked_fill(
                    ~mask, min_value_of_dtype(vector.dtype)
                )
                result = torch.nn.functional.softmax(masked_vector, dim=dim)

        return result

    def forward(self, x, mask):
        batch, seq_len, emb_dim = x.size()

        if self.use_value_mlp:
            value = self.value_mlp(x.view(-1, self.emb_dim))
        else:
            value = x.view(-1, self.emb_dim)

        value = value.view(batch, seq_len, -1)

        att_weights = self.query_mlp(x.view(-1, self.emb_dim))

        att_weights = att_weights.view(batch, seq_len)
        att_weights = self.masked_softmax(
            att_weights, mask, memory_efficient=True
        )
        att_weights = att_weights.unsqueeze(dim=2)

        x = torch.sum(value * att_weights, dim=1)

        return x, att_weights
