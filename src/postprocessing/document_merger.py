import os
from src.corpus.brat_writer import write_file
from typing import Dict, List

import pandas as pd


class DocumentMerger:
    def __init__(
        self,
        ent_id2label,
        rel_id2label,
        true_doc_tokens: Dict[str, List[List[str]]],
        save_dir="val"
    ) -> None:
        super().__init__()
        self.pred_doc_bins = {}
        self.merged_docs = None
        self.true_doc_tokens = true_doc_tokens
        self.ent_id2label = ent_id2label
        self.rel_id2label = rel_id2label
        self.save_dir = save_dir

    def load_partitions(
        self,
        tokens2d: List[List[str]],
        ents: pd.DataFrame,
        rels: pd.DataFrame,
        orig_doc_key: str,
        partition_key: int,
    ):

        doc = dict(
            partition_key=partition_key,
            tokens2d=tokens2d,
            ents=ents,
            rels=rels
        )
        if orig_doc_key in self.pred_doc_bins:
            self.pred_doc_bins[orig_doc_key].append(doc)
        else:
            self.pred_doc_bins[orig_doc_key] = [doc]

    def merge_partitions(self, key, docs):
        """Iteratively, merge document partitions (defined by their predicted 
        `ents` and `rels`) into a doc based on their `orig_doc`. In other words, 
        ents and rels having the same `orig_doc` will be combined.

        Args:
            ents (pd.DataFrame): [description]
            rels (pd.DataFrame): [description]
            doc_key (str): [description]
        """
        tokens2d = []
        entities = pd.DataFrame(columns=['start', 'end', 'step_idx', 'label'])
        relations = pd.DataFrame()
        for d in docs:
            l = int(d['partition_key'])

            tokens2d = tokens2d[:l] + d['tokens2d']

            new_ents = d['ents']

            # adjust step index
            new_ents['step_idx'] = new_ents.apply(
                lambda ent: ent.step_idx + l, axis=1
            )

            # only add entities not found already. ie delete duplicates.
            entities = pd.concat([entities, new_ents]
                                ).drop_duplicates().reset_index(drop=True)

            new_rels = d['rels']
            new_rels['span1'] = new_rels.apply(
                lambda rel: (rel.span1[0], rel.span1[1], rel.span1[2] + l),
                axis=1
            )
            new_rels['span2'] = new_rels.apply(
                lambda rel: (rel.span2[0], rel.span2[1], rel.span2[2] + l),
                axis=1
            )

            relations = pd.concat([relations, new_rels]
                                 ).drop_duplicates().reset_index(drop=True)

        return dict(
            tokens2d=tokens2d,
            entities=entities,
            relations=relations,
            doc_key=key
        )

    def merge(self):
        self.merged_docs = [
           self.merge_partitions(orig_doc_key, doc)
           for orig_doc_key, doc in self.pred_doc_bins.items()
        ] # yapf: disable

        return self.merged_docs

    def save_brat(self):
        for doc in self.merged_docs:
            ents = doc['entities'].copy()
            rels = doc['relations'].copy()

            ents['label'] = doc['entities'].apply(
                lambda ent: self.ent_id2label[ent.label], axis=1
            )
            rels['label'] = doc['relations'].apply(
                lambda rel: self.rel_id2label[rel.label], axis=1
            )
            write_file(
                out_dir=os.path.join(os.getcwd(), self.save_dir),
                name=doc['doc_key'],
                tokens=self.true_doc_tokens[doc['doc_key']],
                ents=ents,
                rels=rels
            )

    def save_prediction(self, format: str):
        if format == 'json':
            self.save_json()
        elif format == 'brat':
            self.save_brat()

    def reset(self):
        self.pred_doc_bins = {}