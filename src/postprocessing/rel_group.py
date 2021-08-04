from enum import Enum
from functools import partial
from typing import Dict, List
from tqdm import tqdm

import pandas as pd


class RelationGroup(Enum):
    # Implicit-Implicit Temporal Relation
    I_I = 0

    # Implicit-Explicit Temporal Relation
    I_E = 1

    # Explicit-Implicit Temporal Relation
    E_I = 2

    # Explicit-Explicit Temporal Relation
    E_E = 3

    # Causal relations
    C = 4

    # Overlap Temporal Relation
    O = 5

    # Within (or intra) Action Phrase Relations
    IAP = 6


# Cross-Action Phrase Temporal and Causal Relations
CAPTAC = [
    RelationGroup.I_I,
    RelationGroup.I_E,
    RelationGroup.E_I,
    RelationGroup.E_E,
    RelationGroup.C,
    RelationGroup.O
]

# Temporal Relations
TEMP = [
    RelationGroup.I_I,
    RelationGroup.I_E,
    RelationGroup.E_I,
    RelationGroup.E_E,
    RelationGroup.O
]

FULL = [
    RelationGroup.I_I,
    RelationGroup.I_E,
    RelationGroup.E_I,
    RelationGroup.E_E,
    RelationGroup.C,
    RelationGroup.O,
    RelationGroup.IAP
]


class RelationGroupFinder:
    def __init__(
        self,
        ents_df: pd.DataFrame,
        rels_df: pd.DataFrame,
        ent_id2label: Dict[int, str],
        rel_id2label: Dict[int, str]
    ):
        self.ent_id2label = ent_id2label
        self.rel_id2label = rel_id2label
        self.ent_label2id = {v: k for k, v in self.ent_id2label.items()}
        self.rel_label2id = {v: k for k, v in self.rel_id2label.items()}
        self.ents_df = ents_df
        self.rels_df = rels_df

    def is_product(self, span):
        partial_cond = (
            (self.rels_df.span2 == span) &
            (self.rels_df.label == self.rel_label2id["Product"])
        )
        return self.rels_df[partial_cond].apply(
            lambda rel: not self.is_action_in_mid(rel.span1, rel.span2), axis=1
        ).any(axis=None)

    def get_entity_type(self, span):
        start, end, step_idx = span
        return self.ents_df[(self.ents_df.start == start) &
                            (self.ents_df.end == end) &
                            (self.ents_df.step_idx == step_idx)].label.values[0]

    def is_action_in_mid(self, head, tail):
        h_s, h_e, h_step = head
        t_s, t_e, t_step = tail

        head_sent_cond = (self.ents_df.step_idx
                          == h_step) & (h_e < self.ents_df.start)
        inbet_sent_cond = (h_step < self.ents_df.step_idx
                          ) & (self.ents_df.step_idx < t_step)
        tail_sent_cond = (self.ents_df.step_idx
                          == t_step) & (self.ents_df.end < t_s)

        if h_step == t_step:
            inbet = head_sent_cond & tail_sent_cond
        else:
            inbet = (head_sent_cond | inbet_sent_cond | tail_sent_cond)

        cond = inbet & (self.ents_df.label == self.ent_label2id["Action"])

        return len(self.ents_df[cond]) > 0

    def rel_group(self, rel: pd.Series):
        head_ent = rel['span1']
        tail_ent = rel['span2']
        rel_type = rel['label']

        head_start, head_end, head_step_idx = head_ent
        tail_start, tail_end, tail_step_idx = tail_ent
        head_type = int(self.get_entity_type(head_ent))
        tail_type = int(self.get_entity_type(tail_ent))

        head_span = (head_start, head_end, head_step_idx)
        tail_span = (tail_start, tail_end, tail_step_idx)

        # 1. implicit implicit: action connected to action using acts-on or site
        if (
            head_type == self.ent_label2id["Action"] and
            tail_type == self.ent_label2id["Action"] and rel_type not in [
                self.rel_label2id[label]
                for label in ["Enables", "Overlaps", "Or"]
            ]
        ):
            return RelationGroup.I_I

        # 2. implicit explicit: action1 connected to an entity in another sentence
        elif (
            head_type == self.ent_label2id["Action"] and
            tail_type != self.ent_label2id["Action"] and
            rel_type == self.rel_label2id["Product"] and
            self.is_action_in_mid(head_span, tail_span)
        ):
            return RelationGroup.I_E

        # 3. Explicit Implicit: action2 connected to an entity in previous sentence
        elif (
            head_type == self.ent_label2id["Action"] and
            tail_type != self.ent_label2id["Action"] and rel_type
            in [self.rel_label2id[label] for label in ["Acts-on", "Site"]] and
            self.is_product(tail_span)
        ):
            return RelationGroup.E_I

        # 4. Explicit Explicit: coreference links between two entities of the two actions
        elif (
            head_type != self.ent_label2id["Action"] and
            tail_type != self.ent_label2id["Action"] and
            rel_type == self.rel_label2id["Coreference-Link"]
        ):
            return RelationGroup.E_E

        # 5. Commands: command links between two actions
        elif (
            head_type == self.ent_label2id["Action"] and
            tail_type == self.ent_label2id["Action"] and
            rel_type == self.rel_label2id["Enables"]
        ):
            return RelationGroup.C

        elif (
            head_type == self.ent_label2id["Action"] and
            tail_type == self.ent_label2id["Action"] and
            rel_type == self.rel_label2id["Overlaps"]
        ):
            return RelationGroup.O

        else:
            # IAP relation
            return RelationGroup.IAP

    # TODO: optimize (128 runs -> 30 min. ~ 13s per run)
    def append_rel_group(self):
        # TODO: break it into 3 steps: 
        # 1. find is_product
        # 2. find is_action_in_mid
        # 3. find rel_group
        #print(len(self.rels_df))
        self.rels_df['group'] = pd.Series(
            [self.rel_group(rel) for rel in self.rels_df.to_dict('records')]
        )
        # append relation group.

        return self.rels_df
