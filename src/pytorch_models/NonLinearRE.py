import itertools as it
import random
from typing import Dict

import numpy as np
import pandas as pd
import torch
from src.corpus.seqeval import get_entities
from src.pytorch_models import device
from src.pytorch_models.mlp import MLP
from src.pytorch_models.s2t_self_attention import SelfAttentionS2T
from torch import nn
from torch.nn import functional as F


class NonLinearRE(nn.Module):
    def __init__(
        self,
        input_emb_size: int,
        label2id: Dict[str, int],
        neg_class: str,
        max_sents: int,
        max_step_gap: int,
        pos_emb_size: int,
        line_emb_size: int
    ):
        super().__init__()
        self.label2id = label2id
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.nb_classes = len(self.label2id)
        self.neg_class_idx = self.label2id[neg_class]
        self.neg_class = neg_class
        self.input_emb_size = input_emb_size
        self.max_sents = max_sents
        self.max_step_gap = max_step_gap
        self.pos_emb_size = pos_emb_size
        self.line_emb_size = line_emb_size
        self.context_dims = 100

        self.fcn = MLP(
            input_features=(
                (self.input_emb_size + self.pos_emb_size) * 2
                + self.line_emb_size + self.context_dims
            ),
            output_features=self.nb_classes,
            arch=[100],
            act="tanh",
            dropout=0.1,
            bias=False
        )
        self.agg_context = SelfAttentionS2T(
            emb_dim=input_emb_size,
            value_out_dim=self.context_dims,
            use_value_mlp=True,
            query_mlp_arch=[],
        )

        self.pos_embeddings = nn.Embedding(
            num_embeddings=self.max_sents + 1, embedding_dim=self.pos_emb_size
        )
        self.line_embeddings = nn.Embedding(
            num_embeddings=self.max_step_gap * 2 + 1,
            embedding_dim=self.line_emb_size,
        )

    def gen_pairs(self, ent_tags):
        chunks = get_entities(ent_tags)
        # filter out special tokens
        chunks = [[ch[1], ch[2]] for ch in chunks if ch[0] != ""]

        # TODO optimize
        pairs = torch.tensor(list(it.product(chunks, repeat=2))).to(device)

        pairs = pairs.view(-1, 4)
        return pairs

    def avg_entities(self, embeddings, rel_pairs):
        """
        use predicted ner bio labels to average the embeddings
        for each named entity.

        return a tensor of named entity embeddings.

        tensor is of size (nb_ner, input_emb_size)
        :param ents: predicted named entities
        :param embeddings:
        :return: tensor([])
        """

        # rel_pairs = [[ent1, ent2]]
        chunks = rel_pairs.reshape(-1, 4)

        # for each chunk, average the embeddings that make up the chunk.
        avg_emb = torch.stack(
            [
                embeddings[s:e + 1].mean(dim=0) for s,
                e,
                _,
                _ in chunks.cpu().tolist()
            ]
        )
        pairs = [
            (pair[1], pair[4]) if pair[1] <= pair[4] else (pair[5], pair[0])
            for pair in rel_pairs.cpu().tolist()
        ]

        max_seq_len = 50
        batch, emb_dims = embeddings.size()

        ctx = torch.stack(
            [
                torch.cat(
                    [
                        embeddings[s:min(e - s, max_seq_len) + s],
                        torch.zeros(
                            (max_seq_len - min(e - s, max_seq_len), emb_dims)
                        ).to(device)
                    ]
                ) for s,
                e in pairs
            ]
        )
        ctx_mask = torch.BoolTensor(
            [
                [True] * min(e - s, max_seq_len) + [False] *
                (max_seq_len - min(e - s, max_seq_len)) for s,
                e in pairs
            ]
        ).to(device)

        # sanity check, avg_emb -> (nb_ne, emb_size)
        assert len(chunks) == avg_emb.size()[0]
        return chunks, avg_emb, ctx, ctx_mask

    def pre_process(self, X, x_rels):
        """Generates a sample of entity pairs based on the fraction of postive
        and negative samples to keep. Returns embeddings for all the filtered
        pairs along with their corresponding truth values (as relation type ids).

        Args:
            X (torch.FloatTensor): [description]
            ent_tags (List[str]): [description]
            true_rels_df (pandas.Dataframe): [description]

        Returns:
            X, ne_pairs: [description]
        """
        seq_len, emb_size = X.size()
        # X Dims: (protocol_seq_len, emb_size)
        # ent_tags list of size: (protocol_seq_len)
        # 1. average the embeddings based on predicted named entities
        ne, X, ctx, ctx_mask = self.avg_entities(X, x_rels)
        # X -> (nb_ents, dims)

        pos_embs = self.pos_embeddings(ne[:, 2])

        nb_ents, dims = ne.size()
        ne_pairs = ne.view(-1, 2 * dims)

        line_embs = self.line_embeddings(ne_pairs[:, 3] - ne_pairs[:, 7] + 5)

        # Take the arg1.end -> arg2.start as context.

        ctx = self.agg_context(ctx, ctx_mask)

        # training only on some positive and negative samples
        X = torch.cat([X, pos_embs], dim=1)

        nb_ents, dims = X.size()
        # X[0] <-> X[1] are pairs.
        X = X.view(-1, 2 * dims)

        X = torch.cat([X, line_embs, ctx], dim=1)
        # X -> (nb_ents/2, emb_size * 2)

        return X, ne_pairs

    def forward(self, tkn_embs, x_rels):
        X, ne_pairs = self.pre_process(tkn_embs, x_rels)

        # (nb_ents * nb_ents, emb_size * 2) ->
        #    (nb_ents * nb_ents, out_dim)
        y_hat = self.fcn(X)

        return y_hat, ne_pairs

    def predict(self, ne_pairs, logits):

        softmax = F.softmax(logits, dim=1)
        labels = torch.argmax(softmax, dim=1)

        id2tag = {v: k for k, v in self.label2id.items()}
        labels = [id2tag[l] for l in labels.cpu().tolist()]

        args = [
            'arg1_start',
            'arg1_end',
            'arg1_sent_idx',
            'arg1_step_idx',
            'arg2_start',
            'arg2_end',
            'arg2_sent_idx',
            'arg2_step_idx'
        ]

        rels_pred_df = pd.DataFrame(ne_pairs.cpu().numpy(),
                                    columns=args).set_index(args)

        rels_pred_df['pred_label'] = labels

        return rels_pred_df

    def gen_y(self, ne_pairs, true_rels):
        # use the entity pairs to generate true rel labels
        ne_pairs_df = ne_pairs.join(true_rels, how="left")
        ne_pairs_df = ne_pairs_df.fillna(self.neg_class)
        y = ne_pairs_df['true_label'].apply(lambda l: self.label2id[l]
                                           ).to_numpy()

        return torch.tensor(y).to(device)

    def loss(self, y_hat, y):
        """
        """
        # flatten all predictions
        ce_loss = 0

        if y_hat.size()[0] == 0:
            # empty prediction
            return ce_loss

        y = y.view(-1)
        y_hat = y_hat.view(-1, self.nb_classes)

        ce_loss = F.cross_entropy(y_hat, y)

        return ce_loss
