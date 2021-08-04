import glob
import json
import os
import random
from collections import namedtuple
from src.pytorch_models import utils
from src.pytorch_models.utils import to_torch_sparse

import pytorch_lightning as pl
import torch
from src.corpus.protocol_dataset import (
    EntitySet, ProtocolDataset, ProtocolSet, RelationSet
)
from src.corpus.tokenizers import WordPieceTokenizer
from torch.utils.data.dataloader import DataLoader

SPECIAL_TOKENS = namedtuple("SPECIAL_TOKENS", ["unk", "pad", "start", "stop"])


class WLPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path,
        dev_path,
        test_path,
        batch_size=1,
        ent_classes=list(),
        ent_neg_class="O",
        num_workers=8,
        max_span_width=10,
        max_pos=100,
        max_step=16,
        max_step_gap=5,
        process_protocols=None,
        per_dev=0,
        rel_classes_set=dict(),
        rel_neg_class="O",
        rel_neg_frac=0.2,
        rel_pos_frac=0.2,
        rel_undirected_classes=list(),
        special_tokens=SPECIAL_TOKENS(
            unk=['UNK'], pad="[PAD]", start="[CLS]", stop="[SEP]"
        ),
        tokenizer=WordPieceTokenizer(),
        use_bert_vocab=True,
        process_titles=False,
    ):
        super().__init__()
        # file paths
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path

        self.batch_size = batch_size
        self.per_dev = per_dev
        self.num_workers = num_workers

        # entity properties
        self.ent_classes = ent_classes
        self.ent_neg_class = ent_neg_class

        self.special_tokens = special_tokens
        self.tokenizer = tokenizer
        self.use_bert_vocab = use_bert_vocab

        # relation properties
        self.max_pos = max_pos
        self.max_step = max_step
        self.max_step_gap = max_step_gap
        self.max_span_width = max_span_width
        self.rel_classes_set = rel_classes_set
        self.nb_rels = len(rel_classes_set["full"]) + 1
        self.rel_neg_class = rel_neg_class
        self.rel_neg_frac = rel_neg_frac
        self.rel_pos_frac = rel_pos_frac
        self.rel_undirected_classes = rel_undirected_classes
        self.process_titles = process_titles

        # For debug purposes. Only process these many protocols
        self.process_protocols = process_protocols

        self.ent_label2id = self.prep_ent_label2id()
        self.ent_id2label = {v: k for k, v in self.ent_label2id.items()}
        self.rel_label2id = self.prep_rel_label2id()
        self.rel_id2label = {v: k for k, v in self.rel_label2id.items()}

    def prep_ent_label2id(self):
        label2id = {k: v + 1 for v, k in enumerate(self.ent_classes)}

        # negative label
        label2id[self.ent_neg_class] = 0

        # needed for batch processing.
        label2id[self.special_tokens.pad] = len(label2id)

        # include special tokens as tags as well
        label2id[self.special_tokens.start] = len(label2id)
        label2id[self.special_tokens.stop] = len(label2id)

        return label2id

    def prep_rel_label2id(self):
        rel_tags = {
            k: v + 1
            for v, k in enumerate(self.rel_classes_set["full"])
        }

        rel_tags[self.rel_neg_class] = 0

        return rel_tags

    def load_filenames(self, dir_path):
        filenames = self.__from_dir(dir_path, extension="ann")
        return filenames

    def setup(self, stage=None):
        """
        Prepare train, dev and test datasets. This function simply puts it 
        all together and returns train dev and test sets.

        * This function does generate a random dev split each time you run this.
        * For reproducibility use a fixed seed for random.
        """

        with open(self.train_path, mode='r') as f:
            json_lines = f.readlines()
            train_protocols = [json.loads(j) for j in json_lines]

        random.shuffle(train_protocols)

        with open(self.dev_path, mode='r') as f:
            json_lines = f.readlines()
            dev_protocols = [json.loads(j) for j in json_lines]

        with open(self.test_path, mode='r') as f:
            json_lines = f.readlines()
            test_protocols = [json.loads(j) for j in json_lines]

        # For debug only, TODO remove
        if self.process_protocols:
            train_protocols = train_protocols[:self.process_protocols]
            dev_protocols = dev_protocols[:self.process_protocols]
            test_protocols = test_protocols[:self.process_protocols]

        print("Building Train Set")
        self.train_db = ProtocolDataset(
            protocols=train_protocols,
            ent_cache=True,
            rel_cache=True,
            vocab=self.tokenizer.vocab(),
            ent_label2id=self.ent_label2id,
            special_tokens=self.special_tokens,
            ent_classes=self.ent_classes,
            max_pos=self.max_pos,
            ent_neg_class=self.ent_neg_class,
            rel_label2id=self.rel_label2id,
            rel_classes=self.rel_classes_set["full"],
            max_step_gap=self.max_step_gap,
            max_span_width=self.max_span_width,
            rel_neg_class=self.rel_neg_class,
            rel_pos_frac=self.rel_pos_frac,
            rel_neg_frac=self.rel_neg_frac,
        )
        print("Building Dev Set")
        self.dev_db = ProtocolDataset(
            protocols=dev_protocols,
            ent_cache=True,
            rel_cache=True,
            vocab=self.train_db.entity_db.vocab,
            ent_label2id=self.ent_label2id,
            special_tokens=self.special_tokens,
            ent_classes=self.ent_classes,
            max_pos=self.max_pos,
            ent_neg_class=self.ent_neg_class,
            max_span_width=self.max_span_width,
            rel_label2id=self.rel_label2id,
            rel_classes=self.rel_classes_set["full"],
            max_step_gap=self.max_step_gap,
            rel_neg_class=self.rel_neg_class,
            rel_pos_frac=1.0,
            rel_neg_frac=1.0,
        )
        print("Building Test Set")
        self.test_db = ProtocolDataset(
            protocols=test_protocols,
            ent_cache=True,
            rel_cache=True,
            vocab=self.train_db.entity_db.vocab,
            ent_label2id=self.ent_label2id,
            special_tokens=self.special_tokens,
            ent_classes=self.ent_classes,
            max_pos=self.max_pos,
            ent_neg_class=self.ent_neg_class,
            max_span_width=self.max_span_width,
            rel_label2id=self.rel_label2id,
            rel_classes=self.rel_classes_set["full"],
            max_step_gap=self.max_step_gap,
            rel_neg_class=self.rel_neg_class,
            rel_pos_frac=1.0,
            rel_neg_frac=1.0,
        )

        self.train_db.log_stats()
        self.dev_db.log_stats()
        self.test_db.log_stats()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_db,
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dev_db,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_db,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def collate_fn(self, sample):
        # use this only if we are processing a single protocol, and ents is
        # a single continuous sequence
        # sanity check
        assert len(sample) == 1, "only 1 sample"
        sample = sample[0]

        # TODO find a cleaner way to tensorify
        return (
            EntitySet(
                tokens2d=sample.ent.tokens2d,
                tkn_ids=torch.tensor(sample.ent.tkn_ids),
                tkn_mask=torch.tensor(sample.ent.tkn_mask),
                spans=torch.tensor(sample.ent.spans),
                span_mask=torch.tensor(sample.ent.span_mask),
                label_ids=torch.tensor(sample.ent.label_ids),
                is_span=torch.FloatTensor(sample.ent.is_span),
                gold_df=sample.ent.gold_df
            ),
            RelationSet(
                indices=torch.tensor(sample.rel.indices),
                label_ids=torch.tensor(sample.rel.label_ids),
                label_mask=torch.tensor(sample.rel.label_mask),
                gold_df=sample.rel.gold_df
            ),
            ProtocolSet(orig_doc=sample.p.orig_doc, part_key=sample.p.part_key)
        )

    @staticmethod
    def __from_dir(folder, extension):
        g = glob.iglob(folder + '/*.' + extension, recursive=True)
        return [os.path.splitext(f)[0] for f in g]
