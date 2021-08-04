import itertools as it
import logging
import math
from collections import Counter, namedtuple
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.coo import coo_matrix
from src.corpus.proto_file import ProtoFile
from src.pytorch_models import utils
from torch.utils import data
from tqdm import tqdm

log = logging.getLogger(__name__)

ProtocolSample = namedtuple('ProtocolSample', ['ent', 'rel', 'p'])

EntitySet = namedtuple(
    "EntitySet",
    [
        "tokens2d",
        "tkn_ids",
        "tkn_mask",
        "spans",
        "span_mask",
        "label_ids",
        "is_span",
        "gold_df"
    ]
)

RelationSet = namedtuple(
    "RelationSet", ["indices", "label_ids", "label_mask", "gold_df"]
)

ProtocolSet = namedtuple("ProtocolSet", ['orig_doc', 'part_key'])


class EntityDataset(data.Dataset):
    def __init__(
        self,
        protocols=None,
        embedding_dim=None,
        pretrained_embed_table=None,
        special_tokens=None,
        max_span_width: int = 10,
        label2id: Dict[str, int] = None,
        split_by_sent: bool = False,
        max_pos: int = 100,
        vocab: Dict[str, int] = None,
        classes: List[str] = None,
        neg_class: str = 'O',
        cache: bool = False,
    ):

        self.protocols = protocols
        self.unk = special_tokens.unk
        self.pad_tkn = special_tokens.pad
        self.start = special_tokens.start
        self.stop = special_tokens.stop
        self.max_pos = max_pos
        self.max_span_width = max_span_width
        self.embedding_dim = embedding_dim
        self.pretrained_embed_table = pretrained_embed_table
        self.label2id = label2id
        self.split_by_sent = split_by_sent
        self.classes = classes
        self.neg_class = neg_class
        self.vocab = vocab
        self.cache = cache

        if self.cache:
            self.data = [
                self.preprocess_protocol(p)
                for p in tqdm(self.protocols, desc="processing entities")
            ]

    def log_stats(self):
        words = []
        labels = []
        tkn_lengths = {}
        for p in self.protocols:
            tokens = list(it.chain.from_iterable(p['tokens']))
            ents = list(it.chain.from_iterable(p['named_entities']))
            ent_labels = [ent['label'] for ent in ents]

            words = words + tokens
            for ent in ents:
                if ent['label'] not in tkn_lengths:
                    tkn_lengths[ent['label']] = Counter()

                tkn_lengths[ent['label']][ent['end'] - ent['start']] += 1

            labels = labels + ent_labels

        word_counts = Counter(words)
        log.info(f"total unique words:{len(word_counts.keys())}")
        log.info(f"total words: {len(words)}")

        log.info(
            "unique words oov: {}".format(
                len([w for w, v in word_counts.items() if w not in self.vocab])
            )
        )

        log.info(
            "#words oov: {}".format(
                len([w for w in words if w not in self.vocab])
            )
        )

        for label, counter in tkn_lengths.items():
            log.info("Entity Length Distribution:")
            log.info(
                f"{label}"
                f"{pd.DataFrame(counter.most_common()).to_markdown()}"
            )

        supp = Counter(labels)
        ent_count = Counter(labels)
        log.info(f"ner support:{supp.most_common()}")
        log.info(f"ner_labels_support:")
        log.info(pd.DataFrame(ent_count.most_common()).to_markdown())
        print(pd.DataFrame(ent_count.most_common()).to_markdown())

    def make_ent_pairs(self, ents):
        ne_pairs = [
            (
                e1['start'],
                e1['end'],
                e1['step_idx'],
                e2['start'],
                e2['end'],
                e2['step_idx']
            ) for (e1, e2) in it.product(ents, repeat=2)
        ]

        return pd.DataFrame(
            ne_pairs,
            columns=[
                'arg1_start',
                'arg1_end',
                'arg1_step_idx',
                'arg2_start',
                'arg2_end',
                'arg2_step_idx'
            ]
        )

    def preprocess_protocol(self, p):

        # convert tokens into a single list of words
        tokens2d = p['tokens']

        # list of words -> list of ids using vocab.
        # if not found, return unk's id.
        tkn_ids = [
            [
                self.vocab[i] if i in self.vocab else self.vocab[self.unk]
                for i in tokens1d
            ]
            for tokens1d in tokens2d
        ]

        tkn_ids, tkn_mask = utils.pad(tkn_ids, self.vocab[self.pad_tkn])

        # get bio tagged labels for each token in the sequence
        ents2d = p['named_entities']

        pos_ents2d = [
            [(e['start'], e['end'], e['label'])
             for e in ents1d]
            for ents1d in ents2d
        ]

        spans2d = [
            utils.enumerate_spans(
                sent_len=len(sent), max_width=self.max_span_width
            ) for sent in tokens2d
        ]

        labels2d = [
            utils.merge_span_labels(spans1d, pos_ents1d)
            for spans1d, pos_ents1d in zip(spans2d, pos_ents2d)
        ] # yapf: disable

        label_ids = [
            [self.label2id[l] for l in labels1d] for labels1d in labels2d
        ]

        spans, span_mask = utils.pad(spans2d, pad=(0, 0))
        label_ids, _ = utils.pad(label_ids, pad=self.label2id[self.pad_tkn])

        neg = self.label2id[self.neg_class]
        pad = self.label2id[self.pad_tkn]
        is_span = [
            [0 if l in [neg, pad] else 1
             for l in labels1d]
            for labels1d in label_ids
        ]

        ents = list(it.chain.from_iterable(p['named_entities']))
        ents = [
            {
                'start': ent['start'],
                'end': ent['end'],
                'step_idx': ent['step_idx'],
                'label': self.label2id[ent['label']]
            } for ent in ents
        ]
        gold_df = pd.DataFrame(
            ents, columns=['start', 'end', 'step_idx', 'label']
        )

        return EntitySet(
            tokens2d=tokens2d,
            tkn_ids=tkn_ids,
            tkn_mask=tkn_mask,
            spans=spans,
            span_mask=span_mask,
            label_ids=label_ids,
            is_span=is_span,
            gold_df=gold_df
        )

    def __getitem__(self, item):
        if self.cache:
            return self.data[item]
        else:
            return self.preprocess_protocol(self.protocols[item])

    def __len__(self):
        return len(self.protocols)


class RelationDataset(data.Dataset):
    def __init__(
        self,
        spans,
        protocols,
        label2id,
        classes,
        neg_class,
        max_span_width,
        max_step_gap,
        pos_frac,
        neg_frac,
        cache=False
    ):
        super().__init__()
        self.gold_rels = [p['relations'] for p in protocols]
        self.label2id = label2id
        self.classes = classes
        self.neg_class = neg_class
        self.max_step_gap = max_step_gap
        self.max_span_width = max_span_width
        self.pos_frac = pos_frac
        self.neg_frac = neg_frac
        self.rels = []
        self.gold_df = None
        self.spans = spans
        self.cache = cache

        if self.cache and len(self.gold_rels) > 0:
            self.rels, self.gold_df = zip(*[
                self.preprocess(s, gold_rels)
                for s, gold_rels in tqdm(
                    zip(self.spans, self.gold_rels),
                    desc="processing relations",
                    total=len(self.gold_rels)
                )
            ]) # yapf: disable

    def preprocess(self, spans2d, gold_rels):
        # from (start, end) -> (start, end, step_idx)
        ents = utils.flatten_spans(spans2d)

        args = [
            'arg1_start',
            'arg1_end',
            'arg1_step_idx',
            'arg2_start',
            'arg2_end',
            'arg2_step_idx'
        ]

        pos_rels = pd.DataFrame(gold_rels, columns=args + ['label'])
        pos_rels = pos_rels.rename(columns={"label": "true_label"})
        # get all entity pairs

        #extract all postive labels for reasonable entity span pairs.
        pos_rels_valid = pos_rels[
            (pos_rels.arg1_end - pos_rels.arg1_start < self.max_span_width) &
            (pos_rels.arg2_end - pos_rels.arg2_start < self.max_span_width)]

        arg1_idx = pos_rels_valid.apply(
            lambda r: ents.index(
                (r['arg1_start'], r['arg1_end'], r['arg1_step_idx'])
            ),
            axis=1
        ).values

        arg2_idx = pos_rels_valid.apply(
            lambda r: ents.index(
                (r['arg2_start'], r['arg2_end'], r['arg2_step_idx'])
            ),
            axis=1
        ).values

        true_label = pos_rels_valid['true_label'].tolist()

        true_label = [self.label2id[l] for l in true_label]

        # initalize rel matrix with negative class as 0
        if len(pos_rels_valid) != 0:
            rels = coo_matrix(
                (true_label, (arg1_idx, arg2_idx)),
                shape=(len(ents), len(ents)),
                dtype=np.int8
            )
        else:
            rels = coo_matrix((len(ents), len(ents)), dtype=np.int8)

        gold_df = pd.DataFrame(
            {
                'span1':
                    list(
                        zip(
                            pos_rels.arg1_start,
                            pos_rels.arg1_end,
                            pos_rels.arg1_step_idx
                        )
                    ),
                'span2':
                    list(
                        zip(
                            pos_rels.arg2_start,
                            pos_rels.arg2_end,
                            pos_rels.arg2_step_idx
                        )
                    ),
                'true_label':
                    pos_rels.true_label
            }
        )

        gold_df.true_label = gold_df.true_label.apply(
            lambda x: self.label2id[x]
        )

        return rels, gold_df

    def gen_relations(self, rels: coo_matrix, pos_frac: int, neg_frac: int):
        """
        Given a list of entity spans (start, end), generate pairwise
        relations.
        """

        # TODO find more e

        # Step 1. Get positive relation indices.
        row, col, data = sparse.find(rels != self.label2id[self.neg_class])
        #//row, col = utils.sparse_all_except(rels, value=self.label2id[self.neg_class])
        pos_idx = np.stack([row, col], axis=-1)

        # Step 2. Get negative relation indices.
        # remove self loops (ie diagonals) from negative samples.
        # TODO optimize SparseEfficienyWarning
        row, col, data = sparse.find(
            (rels == self.label2id[self.neg_class]).multiply(
                ~np.eye(rels.shape[0], dtype=bool)
            )
        )
        neg_idx = np.stack([row, col], axis=-1)

        # Step 3: Randomly sample positive relations
        total = pos_idx.shape[0]
        pos_frac_idx = pos_idx[np.random.choice(
            a=total,
            size=int(pos_frac * total),
            replace=False,
        ), :]

        # Step 4: Randomly sample negative relations
        total = neg_idx.shape[0]
        neg_frac_idx = neg_idx[
            np.random.choice(total, int(neg_frac * total), replace=False), :]

        pair_idx = np.concatenate([pos_frac_idx, neg_frac_idx])

        # convert array(n, 2) -> (array(n), array(n))
        pair_idx = tuple(np.hsplit(pair_idx, 2))
        pair_idx = (
            pair_idx[0].transpose().squeeze(),
            pair_idx[1].transpose().squeeze()
        )
        # Step 5. Make relation matrix and mask
        mask = np.zeros(rels.shape, dtype=bool)
        mask[pair_idx] = True

        return mask

    def log_stats(self):
        pass

    def __getitem__(self, item):
        if self.cache:
            rels = self.rels[item]
            gold_df = self.gold_df[item]
        else:
            rels, gold_df = self.preprocess(
                spans2d=self.spans[item], gold_rels=self.gold_rels[item]
            )

        # convert relations into df
        rel_mask = self.gen_relations(
            rels=rels, pos_frac=self.pos_frac, neg_frac=self.neg_frac
        )

        rows, cols, data = rels.row, rels.col, rels.data

        return RelationSet(
            indices=np.stack([rows, cols], axis=-1),
            label_ids=data,
            label_mask=rel_mask,
            gold_df=gold_df
        )


class ProtocolDataset(data.Dataset):
    def __init__(
        self,
        protocols: List[Dict],
        vocab: Dict[str, int],
        ent_label2id: Dict[str, int],
        special_tokens,
        ent_classes: List[str],
        max_pos: int,
        max_span_width: int,
        ent_neg_class: str,
        rel_label2id: Dict[str, int],
        rel_classes: List[str],
        max_step_gap: int,
        rel_neg_class: str,
        rel_pos_frac: float,
        rel_neg_frac: float,
        ent_cache: bool = False,
        rel_cache: bool = False,
    ):
        super().__init__()
        self.protocols = protocols
        self.rel_neg_class = rel_neg_class

        # skip protocols with no entities and no relations
        self.protocols = [
            p for p in self.protocols
            if len(p['named_entities']) > 0 and len(p['relations']) > 0
        ]

        self.entity_db = EntityDataset(
            protocols=self.protocols,
            vocab=vocab,
            max_span_width=max_span_width,
            label2id=ent_label2id,
            special_tokens=special_tokens,
            classes=ent_classes,
            max_pos=max_pos,
            neg_class=ent_neg_class,
            cache=ent_cache
        )

        spans2d = [
            utils.unpad(data.spans, data.span_mask)
            for data in self.entity_db.data
        ]

        self.relation_db = RelationDataset(
            spans=spans2d,
            protocols=self.protocols,
            label2id=rel_label2id,
            classes=rel_classes,
            neg_class=rel_neg_class,
            max_span_width=max_span_width,
            max_step_gap=max_step_gap,
            pos_frac=rel_pos_frac,
            neg_frac=rel_neg_frac,
            cache=rel_cache
        )

    def log_stats(self):
        self.entity_db.log_stats()
        self.relation_db.log_stats()

    def __getitem__(self, item):
        p = self.protocols[item]
        ent = self.entity_db[item]
        rel = self.relation_db[item]
        # p["doc_key"] = protocol_x_y

        # protocol_x
        orig_doc = "_".join(p["doc_key"].split("_")[:2])

        # y
        part_key = p["doc_key"].split("_")[2]

        return ProtocolSample(
            ent=ent,
            rel=rel,
            p=ProtocolSet(orig_doc=orig_doc, part_key=part_key),
        )

    def __len__(self):
        return len(self.protocols)
