import math
from src.pytorch_models.pos_ffn import posFFN2d
from src.pytorch_models.dropedge import dropedge
from src.pytorch_models.re_rescal import ReRescal

import torch
import torch.nn.functional as F
from src.pytorch_models.mlp import MLP
from torch import nn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

default_gcn_arch = [512, 256, 128]
default_mlp_outs = [384, 256, 128, 64]


class SelfAttentionRE(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.query_mlp = nn.Linear(in_features=emb_dim, out_features=1)
        self.value_mlp = nn.Linear(in_features=emb_dim, out_features=emb_dim)

    def forward(self, x):
        B, N, N, emb_dim = x.shape
        # -> (B x N x N x emb_dim)

        value = self.value_mlp(x.view(-1, emb_dim)).view(B, N, N, emb_dim)
        # -> (B x N x N x emb_dim)

        query = self.query_mlp(x.view(-1, emb_dim)).view(B, N, N, 1)
        # -> (B x N x N x 1)

        att_weights = F.softmax(query, dim=0)
        # -> (B x N x N x 1)

        out = torch.sum(value * att_weights, dim=0)
        # -> (N x N x emb_dim)

        return out, att_weights.detach()


class DecoderIGCN(nn.Module):
    def __init__(
        self,
        nb_rels: int,
        in_features: int = 768,
        rel_emb_dim: int = 100,
        act: str = 'relu',
        n_layers: int = 8,
        dropout=0.2,
        layernorm=False,
        rel_conv: bool = True,
        multi_head_r_gcn: bool = True,
        transcoder_block: bool = True,
    ):
        super().__init__()
        self.nlayers = n_layers

        self.rel_mlp = MLP(
            input_features=rel_emb_dim,
            output_features=nb_rels,
            arch=[],
            act='relu',
            dropout=dropout,
            layernorm=False,
            bias=True
        )
        self.transcoder_block = transcoder_block
        if self.transcoder_block:
            self.igcn = nn.ModuleList(
                [
                    SingleLayerIGCN(
                        nb_rels=nb_rels,
                        rel_decoder=self.rel_mlp,
                        in_features=in_features,
                        rel_emb_dim=rel_emb_dim,
                        dropout=dropout,
                        layernorm=layernorm,
                        window_size=3,
                        rel_conv=rel_conv,
                        multi_head_r_gcn=multi_head_r_gcn
                    ) for i in range(self.nlayers)
                ]
            )

        # self.s2t_re = SelfAttentionRE(nb_rels)
        if not self.transcoder_block:
            self.global_re = ReRescal(
                in_features=in_features,
                out_features=rel_emb_dim,
                dropout=dropout,
                layernorm=layernorm,
            )

    def forward(self, ents, get_extras=False):
        extras = {}
        # Layer 1
        rels = None
        if self.transcoder_block:
            for i in range(self.nlayers):
                ents, rels, extra = self.igcn[i](
                    ents,
                    prev_rels=rels,
                    get_extras=get_extras,
                )
                # -> ents -> (N x dims); rels -> (N x N x d)

                if extra is not None:
                    extras[f'layer_{i}'] = extra

        # # Final Relation extraction Layer
        if not self.transcoder_block:
            rels = self.global_re(ents)

        rels = self.rel_mlp(rels)
        # -> (N x N x r)

        # rels, att = self.s2t_re(torch.stack(adj_list, dim=0))

        if not get_extras:
            extras = None

        return ents, rels, extras


class SingleLayerIGCN(nn.Module):
    def __init__(
        self,
        nb_rels: int,
        in_features: int = 768,
        rel_emb_dim: int = 100,
        rel_decoder: nn.Module = None,
        dropout: float = 0.2,
        dropedge: float = 0.5,
        layernorm: bool = False,
        window_size: int = 3,
        rel_conv: bool = True,
        multi_head_r_gcn: bool = True
    ):
        super().__init__()
        self.dropedge = dropedge
        # TODO add dropout to all rerescal
        if rel_decoder:
            self.rel_mlp = rel_decoder
        else:
            self.rel_mlp = MLP(
                input_features=rel_emb_dim,
                output_features=nb_rels,
                arch=[],
                act='relu',
                dropout=dropout,
                layernorm=False,
                bias=True
            )

        self.local_re = ReRescal(
            in_features=in_features,
            out_features=rel_emb_dim,
            dropout=dropout,
            layernorm=False,
        )
        self.nb_rels = nb_rels

        if multi_head_r_gcn:
            self.multi_head_gcn = MultiHeadGCN(
                nb_heads=nb_rels,
                in_features=in_features,
                arch=[],
                out_features=int(in_features / self.nb_rels),
                dropout=dropout
            )

        self.f_gate_rescal = ReRescal(
            in_features=in_features,
            out_features=nb_rels,
            dropout=dropout,
            layernorm=False,
        )
        self.i_gate_rescal = ReRescal(
            in_features=in_features,
            out_features=nb_rels,
            dropout=dropout,
            layernorm=False,
        )

        self.ent_mlp = MLP(
            input_features=in_features,
            output_features=in_features,
            arch=[],
            act='relu',
            dropout=dropout,
            layernorm=False,
            bias=True
        )
        self.ent_layernorm = nn.LayerNorm(in_features)
        self.rel_layernorm = nn.LayerNorm(rel_emb_dim)

        if rel_conv:
            self.pos_ffn_2d = posFFN2d(
                d_hid=rel_emb_dim,
                d_inner_hid=2 * rel_emb_dim,
                window=window_size,
                dropout=dropout
            )
        self.rel_conv = rel_conv
        self.multi_head_r_gcn = multi_head_r_gcn

    def laplace_smoothing(self, adj):

        # TODO move this outside.
        I = torch.eye(adj.shape[1], device=adj.device)
        I = I.expand(self.nb_rels, -1, -1)
        # -> (r x N x N)

        A_hat = adj + I  # add self-loops
        # -> (r x N x N)

        D_hat_diag = torch.sum(A_hat, dim=2)
        # -> (r x N)

        D_hat_diag_inv_sqrt = torch.pow(D_hat_diag, -0.5)
        # -> (r x N)

        D_hat_diag_inv_sqrt[torch.isinf(D_hat_diag_inv_sqrt)] = 0.
        # -> (r x N)

        D_hat_inv_sqrt = torch.diag_embed(D_hat_diag_inv_sqrt)
        # -> (r x N x N)

        # out = D^-1/2 * A * D^-1/2
        out = torch.bmm(torch.bmm(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)
        # -> (r x N x N)

        return out

    def forward(self, ents, prev_rels=None, get_extras=False):

        residual = ents
        # -> (N, emb_dims)
        N = ents.shape[0]

        extra = {'A': None, 'f_gate': None, 'i_gate': None, 'pred_rels': None}

        rel_embs = F.relu(self.local_re(ents))
        # -> (N x N x d)

        # res connection
        if prev_rels is not None:
            rel_embs = self.rel_layernorm(rel_embs + prev_rels)
        # -> (N x N x d)

        if self.rel_conv:
            rel_embs = self.pos_ffn_2d(rel_embs)
        # -> (N x N x d)

        iap_rels = self.rel_mlp(rel_embs)
        # -> (N x N x r)

        iap_rels = iap_rels.permute(2, 0, 1)
        # -> (r x N x N)

        # if prev_rels is not None:
        #     f_gate = F.sigmoid(self.f_gate_rescal(ents)).permute(2, 0, 1)
        #     # -> (r x N x N)
        #     i_gate = F.sigmoid(self.i_gate_rescal(ents)).permute(2, 0, 1)
        #     # -> (r x N x N)

        #     if get_extras:
        #         extra['f_gate'] = f_gate.detach()
        #         extra['i_gate'] = i_gate.detach()

        #     iap_rels = f_gate * prev_rels + i_gate * iap_rels
        # else:
        #     i_gate = F.sigmoid(self.i_gate_rescal(ents)).permute(2, 0, 1)
        #     # -> (r x N x N)
        #     if get_extras:
        #         extra['i_gate'] = i_gate.detach()

        #     iap_rels = i_gate * iap_rels

        if get_extras:
            extra['pred_rels'] = F.softmax(iap_rels, dim=0).detach()

        A = F.softmax(iap_rels, dim=0)
        # -> (r x N x N)

        # delete negative relation
        # //A = A[:, :, 1:]
        # // -> (N, N, r-1)

        # normalize before GCN pass
        A = self.laplace_smoothing(A)
        # -> (r x N x N)

        A = dropedge(A, p=self.dropedge, training=self.training)

        if get_extras:
            extra['A'] = A.detach()

        ents_slices = ents.expand(self.nb_rels, -1, -1)
        # -> (r, N, emb_dims)

        if self.multi_head_r_gcn:
            ents = self.multi_head_gcn(ents_slices, A)
            # -> (r x N x out_dims); out_dims = emb_dims/r

            # concat on the first dim
            ents = ents.permute(1, 0, 2).reshape(N, -1)
            # -> (N x emb_dims)

        ents = self.ent_mlp(ents)

        ents = self.ent_layernorm(ents + residual)

        if not get_extras:
            extra = None

        return ents, rel_embs, extra


class MultiHeadGCN(nn.Module):
    def __init__(self, in_features, arch, out_features, nb_heads, dropout):
        super().__init__()
        self.nb_heads = nb_heads
        self.arch = [in_features] + arch + [out_features]
        self.gc_layer = nn.ModuleList(
            [
                GraphConvolutionDirected(
                    self.nb_heads, self.arch[i], self.arch[i + 1]
                ) for i in range(len(self.arch) - 1)
            ]
        )

        self.dropout = dropout

    def forward(self, x, adj):
        # -> x -> (B x N x dims)
        # -> adj -> (B x N x N)
        for i in range(len(self.arch) - 1):
            x = F.relu(self.gc_layer[i](x, adj))
            x = F.dropout(x, self.dropout, training=self.training)

        return x


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, nb_heads, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nb_heads = nb_heads
        self.weight = Parameter(
            torch.FloatTensor(nb_heads, in_features, out_features)
        )
        if bias:
            self.bias = Parameter(torch.FloatTensor(nb_heads, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        output = torch.bmm(adj, torch.bmm(input, self.weight))
        if self.bias is not None:
            return output + self.bias.unsqueeze(1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GraphConvolutionDirected(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, nb_heads, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nb_heads = nb_heads
        self.weight_f = Parameter(
            torch.FloatTensor(nb_heads, in_features, out_features)
        )
        self.weight_b = Parameter(
            torch.FloatTensor(nb_heads, in_features, out_features)
        )
        if bias:
            self.bias = Parameter(torch.FloatTensor(nb_heads, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_f.size(1))
        self.weight_f.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight_b.size(1))
        self.weight_b.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # -> input -> (B x N x dims)
        # -> adj -> (B x N x N)
        adj_f, adj_b = adj, adj.transpose(1, 2)
        out_f = torch.bmm(adj_f, torch.bmm(input, self.weight_f))
        out_b = torch.bmm(adj_b, torch.bmm(input, self.weight_b))
        if self.bias is not None:
            return out_f + out_b + self.bias.unsqueeze(1)
        else:
            return out_f + out_b

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
