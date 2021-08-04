from enum import Enum
from functools import partial
from typing import Dict, List

import pandas as pd
from pyspark import sql
from pyspark.sql import functions as F
from pyspark.sql.functions import monotonically_increasing_id, udf, when
from pyspark.sql.types import BooleanType, IntegerType
from src.spark import py_or_udf, start_spark
from tqdm import tqdm


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

    def __int__(self):
        return self.value


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
        ent_id2label: Dict[int, str] = None,
        rel_id2label: Dict[int, str] = None
    ):
        self.ent_id2label = ent_id2label
        self.rel_id2label = rel_id2label
        if self.ent_id2label is not None:
            self.ent_label2id = {v: k for k, v in self.ent_id2label.items()}

        if self.rel_id2label is not None:
            self.rel_label2id = {v: k for k, v in self.rel_id2label.items()}

    def get_entity_type(self, span):
        start, end, step_idx = span
        return self.ents_df[(self.ents_df.start == start) &
                            (self.ents_df.end == end) &
                            (self.ents_df.step_idx == step_idx)].label.values[0]

    def is_true_product(self, rels_sdf: sql.DataFrame):
        # A relation is a true product if the reagent that is being produced
        # is in its own action phrase.

        # eg: Action1 ---Product---> Reagent is a true product.
        # but not
        # 1. Action1 ----Product ------↓
        # 2. Action2 ----Acts-on---> Reagent

        rels_sdf = rels_sdf.withColumn(
            "is_true_product",
            when(
                (rels_sdf['label'] == "Product") &
                (~rels_sdf['is_action_in_mid']),
                True
            ).otherwise(False)
        )
        return rels_sdf

    def is_action_in_mid(
        self, ents_sdf: sql.DataFrame, rels_sdf: sql.DataFrame
    ):
        # check if there is an action in the between the two entities
        # of a relation.

        # TODO join on condition
        # f_udf = udf(ufunc, BooleanType())
        # rels_sdf = rels_sdf.withColumn(
        #     "is_action_in_mid",
        #     f_udf(
        #         rels_sdf['arg1_start'],
        #         rels_sdf['arg1_end'],
        #         rels_sdf['arg1_step_idx'],
        #         rels_sdf['arg2_start'],
        #         rels_sdf['arg2_end'],
        #         rels_sdf['arg2_step_idx'],
        #         ents_sdf
        #     )
        # )

        head_sent_cond = (ents_sdf.step_idx == rels_sdf.arg1_step_idx
                         ) & (rels_sdf.arg1_end < ents_sdf.start)
        inbet_sent_cond = (rels_sdf.arg1_step_idx < ents_sdf.step_idx
                          ) & (ents_sdf.step_idx < rels_sdf.arg2_step_idx)
        tail_sent_cond = (ents_sdf.step_idx == rels_sdf.arg2_step_idx
                         ) & (ents_sdf.end < rels_sdf.arg2_start)
        inbet = (
            (rels_sdf.arg1_step_idx == rels_sdf.arg2_step_idx) &
            (head_sent_cond & tail_sent_cond)
        ) | (
            (rels_sdf.arg1_step_idx != rels_sdf.arg2_step_idx) &
            (head_sent_cond | inbet_sent_cond | tail_sent_cond)
        )

        cond = inbet & (ents_sdf.ent_label == "Action")
        rels_ents_sdf = rels_sdf.join(ents_sdf, cond, 'outer')

        rels_ents_sdf = rels_ents_sdf.groupBy('rid').agg(
            F.collect_list("eid").alias("ents_list")
        )
        rels_ents_sdf = rels_ents_sdf.withColumn(
            'is_action_in_mid',
            when(F.size(rels_ents_sdf['ents_list']) > 0, True).otherwise(False)
        )

        rels_sdf = rels_sdf.join(rels_ents_sdf, on='rid', how='left')

        return rels_sdf

    def rel_group(self, rels_sdf: sql.DataFrame):
        def ufunc(
            head_type, tail_type, rel_type, is_true_product, is_action_in_mid
        ):

            # 1. implicit implicit: action connected to action using acts-on or site
            if (
                head_type == "Action" and tail_type == "Action" and
                rel_type not in ["Enables", "Overlaps", "Or"]
            ):
                return int(RelationGroup.I_I)

            # 2. implicit explicit: action1 connected to an entity in another
            # sentence
            elif (
                head_type == "Action" and tail_type != "Action" and
                rel_type == "Product" and is_action_in_mid
            ):
                return int(RelationGroup.I_E)

            # 3. Explicit Implicit: action2 connected to an entity in previous
            # sentence
            elif (
                head_type == "Action" and tail_type != "Action" and
                rel_type in ["Acts-on", "Site"] and is_true_product
            ):
                return int(RelationGroup.E_I)

            # 4. Explicit Explicit: coreference links between two entities of the
            # two actions
            elif (
                head_type != "Action" and tail_type != "Action" and
                rel_type == "Coreference-Link"
            ):
                return int(RelationGroup.E_E)

            # 5. Commands: command links between two actions
            elif (
                head_type == "Action" and tail_type == "Action" and
                rel_type == "Enables"
            ):
                return int(RelationGroup.C)

            elif (
                head_type == "Action" and tail_type == "Action" and
                rel_type == "Overlaps"
            ):
                return int(RelationGroup.O)

            else:
                # IAP relation
                return int(RelationGroup.IAP)

        f = udf(ufunc, IntegerType())
        rels_sdf = rels_sdf.withColumn(
            "group",
            f(
                rels_sdf['ent_label_arg1'],
                rels_sdf['ent_label_arg2'],
                rels_sdf['label'],
                rels_sdf['is_true_product'],
                rels_sdf['is_action_in_mid']
            )
        )

        return rels_sdf

    # TODO: optimize (128 runs -> 30 min. ~ 13s per run)
    def append_rel_group(
        self,
        ents_df: pd.DataFrame,
        rels_df: pd.DataFrame,
    ):
        sp = start_spark()
        ents_sdf = sp.createDataFrame(ents_df)
        rels_sdf = sp.createDataFrame(rels_df)

        # 0. add eid
        ents_sdf = ents_sdf.withColumn("eid", monotonically_increasing_id())
        rels_sdf = rels_sdf.withColumn("rid", monotonically_increasing_id())

        # 1. find is_action_in_mid
        rels_sdf = self.is_action_in_mid(ents_sdf, rels_sdf)

        # 2. find is_true_product
        rels_sdf = self.is_true_product(rels_sdf)

        # 3. find rel_group
        rels_sdf = self.rel_group(rels_sdf)

        return rels_sdf.toPandas()
