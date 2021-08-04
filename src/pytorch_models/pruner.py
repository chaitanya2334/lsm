from typing import Union

import torch
from src.pytorch_models import utils
from torch import BoolTensor, FloatTensor, LongTensor, nn


# TODO cleanup
class Pruner(nn.Module):
    def __init__(
        self,
        scorer: nn.Module = None,
        pad_id: int = 18,
        neg_sample_id: int = 0,
        gold_beam: bool = False,
        min_score: float = None,
    ):
        super().__init__()
        self.scorer = scorer
        self.gold_beam = gold_beam
        self.min_score = min_score
        self.pad_id = pad_id
        self.neg_sample_id = neg_sample_id

    def forward(
        self,
        span_embs: FloatTensor,
        span_mask: LongTensor,
        max_spans: Union[int, LongTensor],
        scores: FloatTensor,
        gold_labels: LongTensor = None
    ) -> BoolTensor:
        """Return top-n scoring spans. If gold_labels are passed, return only 
        those spans.

        Args:
            span_embs (FloatTensor): 3d tensor of size 
                (batch_size, max_n_spans, emb_dim) 
            containing span representations that will be filtered based on 
                their scores If the scores are passed this is not needed.
            span_mask (LongTensor): 2d tensor of size (batch_size, max_n_spans) 
                marking spans that are unpadded.
            max_spans (Union[int, LongTensor]): a scalar to specify a cap on 
                how many spans to keep.
            scores (FloatTensor): 2d tensor of size 
                (batch_size, max_n_spans) softmax scores for each span.
            gold_labels (LongTensor): 2d tensor of size 
                (batch_size, max_n_spans) label ids for each span.

        Returns:
            BoolTensor: A mask to filter out low scoring spans, and keep 
                top-n spans.
        """

        # make max_spans of size (batch_size) if passed as a scalar
        if isinstance(max_spans, int):
            max_spans = torch.ones(span_mask.shape[0]) * max_spans
            max_spans = max_spans.long()
            # -> (batch_size)

        if gold_labels is not None:
            scores = torch.where(
                (gold_labels != self.pad_id) &
                (gold_labels != self.neg_sample_id),
                torch.zeros_like(gold_labels, dtype=torch.float),
                utils.NEG_INF * torch.ones_like(gold_labels, dtype=torch.float)
            )
            # -> (batch_size, max_n_spans, 1)
            max_spans = torch.sum(
                (gold_labels != self.pad_id) &
                (gold_labels != self.neg_sample_id),
                dim=1
            )
            self.min_score = 0

        elif scores is None:
            # generate scores using the scorer else use the passed scores
            scores = self.scorer(span_embs)
            # make sure the scores for padded spans are not considered.
            scores = scores * span_mask

        if self.min_score is not None:
            # -> scores -> (batch_size, max_n_spans)
            nb_high_scores = torch.sum(scores >= self.min_score, dim=1)
            # -> (batch_size)

        # make sure you get atleast one span per sentence.
        max_spans = torch.maximum(
            torch.min(max_spans, nb_high_scores), torch.ones_like(max_spans)
        )

        assert all(max_spans >= 1)

        # -> scores -> (batch_size, max_n_spans)
        top_scores, top_idx, top_mask = utils.vectorized_topk(
            scores=scores,
            k=max_spans,
            dim=1,
        )
        # -> top_scores -> (batch_size, max_spans_to_keep)
        # -> top_idx -> (batch_size, max_spans_to_keep)
        # -> top_mask -> (batch_size, max_spans_to_keep)

        top_span_mask = torch.zeros_like(span_mask)

        # fill top_idx padded pos with one of the valid index.
        fill_value, _ = top_idx.max(dim=1)
        fill_value = fill_value.unsqueeze(-1)
        top_idx = torch.where(top_mask, top_idx, fill_value)
        top_span_mask = top_span_mask.scatter(1, top_idx, 1)

        # make sure top_span_mask does not include masked out entries in span_mask
        top_span_mask = top_span_mask * span_mask

        return top_span_mask
