import gc
from typing import List

import torch
from src.pytorch_models.mlp import MLP
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForTokenClassification


class Bert(nn.Module):
    def __init__(self, hidden_dim, nb_ents):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nb_ents = nb_ents

        # get pretrained bioBERT model
        self.bert = AutoModelForTokenClassification.from_pretrained(
            "dmis-lab/biobert-v1.1",
            return_dict=True,
            output_hidden_states=True,
            num_labels=self.nb_ents
        )

    def forward(self, input_ids, attention_mask):
        # batchify bert for large documents
        bert_out = self.bert(input_ids, attention_mask=attention_mask)
        token_embeddings = bert_out.hidden_states[12]
        # token_embeddings -> (batch, seq_len, hidden_dims)

        cls_tokens = token_embeddings[:, 0, :].squeeze(1)
        # cls_tokens -> (batch, hidden_dims)

        return token_embeddings, cls_tokens

    def predict(self, logits, offsets) -> List[str]:
        protocol_seq_len, _ = logits.size()

        # get predictions.
        softmax = F.softmax(logits, dim=1)
        ner_pred = torch.argmax(softmax, dim=1)

        bio_ne = ner_pred.cpu().tolist()

        bio_ne = [self.id2label[ne] for ne in bio_ne]
        # convert special tokens to "O". # TODO fix hardcoded special tokens
        bio_ne = [
            'O' if ne in ['[SEP]', '[CLS]', '[PAD]'] else ne for ne in bio_ne
        ]

        # split into sentences.
        bio_ne_2d = [
            bio_ne[offsets[i]:offsets[i + 1]]
            for i in range(offsets.size(0) - 1)
        ]

        return bio_ne_2d

    def loss(self, y_hat, y, attention_mask):
        # y_hat -> (batch, seq_len, nb_ents)
        # y -> (batch, seq_len)

        # remove the padding, its causing NaN in logSoftmax
        # y_hat = y_hat[attention_mask > 0]
        y = y[attention_mask > 0]
        # y_hat -> (protocol_seq_len, nb_ents)
        # y -> (protocol_seq_len)

        # flatten all predictions
        y = y.view(-1)
        # y -> (batch*seq_len)
        y_hat = y_hat.view(-1, self.nb_ents)
        # y_hat -> (batch*seq_len, nb_ents)

        return F.cross_entropy(y_hat, y)
