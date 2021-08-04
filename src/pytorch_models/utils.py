from typing import Any, List, Tuple

import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from scipy import sparse
import torch
import torch.nn.functional as F
from scipy.sparse.coo import coo_matrix
from torch import BoolTensor, LongTensor, Tensor

NEG_INF = -1e20


def pool_embeddings(embeddings, intervals, mode, device):
    if len(intervals) == 0:
        raise ValueError("no entities found")
    input = torch.cat([torch.arange(s, e + 1) for s, e in intervals]).to(device)
    offsets = torch.cumsum(
        torch.Tensor([0] + [e - s + 1 for s, e in intervals[:-1]]), dim=0
    ).long().to(device)

    return F.embedding_bag(
        weight=embeddings,
        input=input,
        offsets=offsets,
        mode=mode,
    )


def batched_idx_select(target: Tensor, idx: LongTensor):
    """
    Given target of size (batch_size, max_seq_len, emb_dim), extract embedding
    for each index in `idx`. The `idx` has indices relevant to each sequence in 
    the batch such that the size is (batch_size, num_indices). Therefore must 
    extract embedding from each sequence in the batch corresponding to the 
    listed indicies for that batch.

    Args:
        target (Tensor): 3d tensor of size (batch_size, max_seq_len, emb_dim) 
            from which to extract embeddings based on indices provided in `idx`.
        idx (LongTensor): 2d tensor of size (batch_size, num_indices) containing
            a list of indicies to be extracted for each sequence in the 
            batched `target`.
    """
    dummy = idx.unsqueeze(2).expand(idx.size(0), idx.size(1), target.size(2))
    # index -> (batch_size, num_indices, emb_dim)
    out = target.gather(1, dummy)
    # out -> (batch_size, num_indices, emb_dim)
    return out


def batched_span_select(target: Tensor, spans: LongTensor):
    """
    
    Given target of size (batch_size, max_seq_len, emb_dim), extract span 
    embeddings defined by indices in `spans`.


    Args:
        target (Tensor): 3d tensor of size (batch_size, max_seq_len, emb_dim) 
            from which to extract embeddings
        spans (LongTensor): 3d tensor of size (batch_size, num_spans, 2) 
            containing pair of indices for each span to be extracted in each 
            sequence in the batch.
    
    Returns:
        span_embeddings (Tensor): a 4d tensor of size (batch_size, num_spans, 
        max_span_len, emb_dim) containing token embeddings defined for each 
        span, padded to `max_span_len`.
        mask (BoolTensor): a 3d tensor of size (batch_size, num_spans, max_width)
            indicating the embeddings to take from span_embeddings for each 
            span in each batch.
    """
    start, end = spans.split(1, dim=-1)
    # start -> (batch_size, num_spans, 1)
    # end -> (batch_size, num_spans, 1)

    widths = end - start
    max_width = widths.max() + 1

    # * imp: setting device is needed.
    range_idx = torch.arange(
        0, max_width, dtype=torch.long, device=start.device
    )
    # span_range_idx -> (max_width)

    # s, s+1, s+2, ... s + max_width
    idx = start + range_idx.view(1, 1, -1)
    # idx -> (batch_size, num_spans, max_width)
    idx_shape = idx.shape

    mask = (
        # s, s+1, s+2, ... e, 0, 0, 0
        (range_idx.view(1, 1, -1) <= widths) &
        # sanity check. no index out of bounds.
        (0 <= idx) &
        # sanity check. no index out of bounds.
        (idx < target.size(1))
    )

    # apply mask to only look for valid indices
    idx = idx * mask  # s, s+1, s+2, ... e, 0, 0, 0
    # idx -> (batch_size, num_spans, max_width)

    # flatten idx
    idx = idx.view(idx.size(0), -1)
    # idx -> (batch_size, num_spans * max_width)

    span_embeddings = batched_idx_select(target, idx)
    # span_embeddings -> (batch_size, num_spans * max_width, emb_dim)

    span_embeddings = span_embeddings.view(*idx_shape, -1)
    # span_embeddings -> (batch_size, num_spans, max_width, emb_dim)

    # mask -> (batch_size, num_spans, max_width)

    return span_embeddings, mask


def enumerate_spans(sent_len: int,
                    max_width: int = None,
                    min_width: int = 0) -> List[Tuple[int, int]]:
    max_width = max_width or sent_len

    spans = [
        (start, end) for start in range(sent_len) for end in range(
            min(start + min_width, sent_len), min(start + max_width, sent_len)
        )
    ]

    return spans


def merge_span_labels(
    spans: List[Tuple[int, int]],
    ents: List[Tuple[int, int, str]],
    neg_class: str = "O"
) -> List[str]:
    spans_df = pd.DataFrame(spans, columns=["s", "e"])
    ents_df = pd.DataFrame(ents, columns=["s", "e", "l"])

    df = pd.merge(spans_df, ents_df, how="left", on=["s", "e"])
    df = df.fillna(neg_class)
    return df["l"].values.tolist()


def pad(x: List[List[Any]],
        pad: int) -> Tuple[List[List[int]], List[List[bool]]]:
    """Pad 2d list x based on max length. Also generate a mask to access valid
        values in the padded list.

        Args:
            x (List[List[Any]]): The 2d list of values to be padded with pad 
                values.
            pad (Any): the value that will be used to pad x.

        Returns:
            Tuple[List[List[int]], List[List[bool]]]: padded x along with its 
                mask 
        """

    max_length = max(len(sample) for sample in x)

    mask = [
        [True] * len(sample) + [False] * (max_length - len(sample))
        for sample in x
    ]

    x = [sample + [pad] * (max_length - len(sample)) for sample in x]

    return x, mask


def flatten_spans(
    spans: List[List[Tuple[int, int]]]
) -> List[Tuple[int, int, int]]:
    """
    Convert 2d list of spans where each span is a tuple (start, end) to a
    flattened reprsentation where each span is a tuple (start, end, step_idx)
    """
    return [
        (*s, i) for i, spans_in_step in enumerate(spans) for s in spans_in_step
    ]


def deflate_spans(
    spans: List[Tuple[int, int, int]]
) -> List[List[Tuple[int, int]]]:
    """[summary]

    Args:
        spans (List[Tuple[int, int, int]]): [description]

    Returns:
        List[List[Tuple[int, int]]]: [description]
    """
    # assuming last element in the tuple is the step id
    spans = sorted(spans, key=lambda x: (x[2], x[0], x[1]))

    max_step = max(step_idx for _, _, step_idx in spans)

    return [
        [(s, e)
         for s, e, step_idx in spans
         if step_idx == idx]
        for idx in range(max_step + 1)
    ]


def unpad(x, mask, row_modifier=None):
    if row_modifier is not None:
        return [
            row_modifier([r for i, r in enumerate(row) if row_mask[i]])
            for row, row_mask in zip(x, mask)
        ] # yapf: disable
    else:
        return [
            [r for i, r in enumerate(row) if row_mask[i]]
            for row, row_mask in zip(x, mask)
        ] # yapf: disable


def to_torch_sparse(coo: coo_matrix):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def to_scipy_sparse(x: torch.sparse.FloatTensor):
    raise NotImplementedError


def vectorized_topk(scores: torch.Tensor, k: torch.Tensor, dim: int):
    """Return top k scores for every k in the given dimension.

    Args:
        x (torch.Tensor): Scores to be used
        k (torch.Tensor): k is of size (batch_size)
        dim (int): Across this dimension
    """
    max_n = int(max(k.max().item(), 1))

    top_score, top_score_idx = torch.topk(scores, max_n, dim=dim)
    mask = k.unsqueeze(dim) >= k.new_ones(k.shape[0], max_n).cumsum(dim=dim)
    top_score_idx = torch.where(
        mask,
        top_score_idx,
        torch.iinfo(torch.long).min * torch.ones_like(top_score_idx)
    )
    top_score = torch.where(
        mask,
        top_score,
        torch.finfo(torch.float32).min * torch.ones_like(top_score)
    )

    return top_score, top_score_idx, mask


def make_pairs_by_mask(mask):
    """Given a 1d `mask` of size (n), generate a 2d tensor `out_mask` of 
    size (n, n) such that `out_mask[i, j] = mask[i] and mask[j]`.

    Args:
        mask (BoolTensor): 1d tensor of size n.
    Returns:
        out_mask (BoolTensor): 2d tensor of size (n, n) where n is the dim of 
            `mask`.
    """
    # cast from bool to uint8 and expand on 2nd dim
    mask_int = mask.float().unsqueeze(1)
    out_mask = mask_int.mm(mask_int.transpose(0, 1))

    # cast back to bool.
    return out_mask.bool()


def make_pairs_by_sent_mask(mask, sent_mask):
    """Given a 2d mask (s x n) generate a pairwise mask of size (sn x sn) such 
    that each row in 2d mask pairs with each others

    Args:
        mask ([type]): [description]
    """
    pairs = make_pairs_by_mask(mask.view(-1))
    sent_lengths = torch.sum(sent_mask, dim=1).tolist()

    sent_mask = torch.block_diag(
        *[torch.ones(s, s).to(mask.device) for s in sent_lengths]
    )

    pairs = pairs * sent_mask

    return pairs.bool()


def tensor_flatten_spans(spans: Tensor, span_mask: BoolTensor):
    # spans -> (batch_size, n_spans, 2)
    # span_mask -> (batch_size, n_spans)
    step = torch.ones_like(span_mask).nonzero()[:, 0]
    # step -> (batch_size * n_spans)
    step = step.view(spans.shape[:2]).unsqueeze(2)
    # step -> (batch_size, n_spans, 1)
    spans = torch.cat([spans, step], dim=2)
    # spans -> (batch_size, n_spans, 3)
    return spans[span_mask]
    # ret -> (valid_spans, 3)


def sparse_all_except(x, value):
    nonzero_mask = np.array(x[x.nonzero()] != value)[0]
    rows = x.nonzero()[0][nonzero_mask]
    cols = x.nonzero()[1][nonzero_mask]
    return rows, cols


def sparse_find(x, value):
    nonzero_mask = np.array(x[x.nonzero()] == value)[0]
    rows = x.nonzero()[0][nonzero_mask]
    cols = x.nonzero()[1][nonzero_mask]
    return rows, cols


def sparse_random(x, n):
    nrows, ncols = x.shape
    density = n / (nrows * ncols - x.count_nonzero())
    W = x.copy()
    W.data[:] = 1
    W2 = sparse.csr_matrix((nrows, ncols))
    while W2.count_nonzero() < n:
        W2 += sparse.random(nrows, ncols, density=density, format='csr')
        # remove nonzero values from W2 where W is 1
        W2 -= W2.multiply(W)
    W2 = W2.tocoo()
    r = W2.row[:n]
    c = W2.col[:n]

    return r, c
