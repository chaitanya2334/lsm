import itertools as it
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pytorch_lightning.metrics.metric import Metric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src import utils as sutils
from src.postprocessing.document_merger import DocumentMerger
from src.postprocessing.rel_group import (CAPTAC, FULL, TEMP, RelationGroup,
                                          RelationGroupFinder)
from src.postprocessing.tensorboard import plot_confusion_matrix
from tqdm import tqdm


class EvalWLP(Metric):
    def __init__(
        self, classes, id2label, neg_class, prefix, compute_on_step=False
    ):
        super().__init__(compute_on_step=compute_on_step)
        self.classes = classes
        self.prefix = prefix
        self.id2label = id2label
        self.neg_class = neg_class
        label2id = {v: k for k, v in self.id2label.items()}
        self.neg_class_id = label2id[neg_class]

        self.add_state('pred', default=[])
        self.add_state('true', default=[])

    def prf_table(self):
        return

    def per_class_table(self):
        return

    def compute(self):
        preds = np.concatenate(self.pred)
        target = np.concatenate(self.true)

        preds = [self.id2label[t] for t in preds.tolist()]
        target = [self.id2label[t] for t in target.tolist()]
        p, r, f1, _ = precision_recall_fscore_support(
            y_pred=preds,
            y_true=target,
            average='micro',
            labels=self.classes
        )
        acc = accuracy_score(y_pred=preds, y_true=target)
        p_class, r_class, f1_class, _ = precision_recall_fscore_support(
            y_true=target,
            y_pred=preds,
            average=None,
            labels=self.classes
        )
        image = plot_confusion_matrix(
            true=target,
            pred=preds,
            classes=self.classes + [self.neg_class],
            title=f'Confusion matrix for {self.prefix}'
        )

        tables = {
            "prf":
                pd.DataFrame({
                    "P": [p], "R": [r], "F1": [f1]
                }),
            "per_class":
                pd.DataFrame(
                    {
                        "P": p_class, "R": r_class, "F1": f1_class
                    },
                    index=self.classes
                )
        }

        metrics = sutils.flatten_dict(
            {
                'prf': {
                    'Acc': acc, 'P': p, 'R': r, 'F1': f1
                },
                "per_class":
                    {
                        label: {
                            "P": p_class[i],
                            "R": r_class[i],
                            "F1": f1_class[i],
                        }
                        for i,
                        label in enumerate(self.classes)
                    }
            }
        )

        return metrics, tables, image


class EvalEntWLP(EvalWLP):
    def __init__(self, classes, id2label, neg_class, prefix):
        super().__init__(
            classes=classes,
            id2label=id2label,
            neg_class=neg_class,
            prefix=prefix
        )

    def update(self, pred_df, true_df) -> None:
        merge = pred_df.set_index(['start', 'end', 'step_idx']).join(
            true_df.set_index(['start', 'end', 'step_idx']),
            how='outer',
            lsuffix="_pred",
            rsuffix="_true"
        ).fillna(self.neg_class_id)
        self.pred.append(merge.label_pred.values.astype(int))
        self.true.append(merge.label_true.values.astype(int))


class EvalRelWLP(EvalWLP):
    def __init__(
        self,
        classes,
        id2label,
        neg_class,
        prefix,
        keep_groups: List[RelationGroup],
        keep_gaps: List[int] = None,
    ):
        super().__init__(
            classes=classes,
            id2label=id2label,
            neg_class=neg_class,
            prefix=prefix
        )

        self.keep_groups = keep_groups
        self.keep_gaps = keep_gaps

    def update(self, pred_df, true_df) -> None:
        # update rels
        pred_df = self.filter_rels(pred_df)
        true_df = self.filter_rels(true_df)

        merged = pred_df.set_index(['span1', 'span2']).join(
            true_df.set_index(['span1', 'span2']),
            how='outer',
            lsuffix="_pred",
            rsuffix="_true"
        ).fillna(self.neg_class_id)

        pred_labels = merged.label_pred.values.astype(int)
        true_labels = merged.label_true.values.astype(int)

        self.pred.append(pred_labels)
        self.true.append(true_labels)

    def filter_rels(self, rels_df):

        is_groups = rels_df.apply(
            lambda rel: rel.group in self.keep_groups, axis=1
        )
        cond = is_groups

        if self.keep_gaps is not None:
            is_gaps = rels_df.apply(
                lambda rel: abs(rel.span1[2] - rel.span2[2]) in self.keep_gaps,
                axis=1
            )
            cond = cond & is_gaps

        rels_df = rels_df[cond]

        return rels_df


class MetaEvalWLP:
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.update(*args, **kwds)

    def __init__(
        self,
        ent_classes: List[str],
        rel_classes_set: Dict[str, List[str]],
        ent_id2label: Dict[int, str],
        rel_id2label: Dict[int, str],
        ent_neg_class: str,
        rel_neg_class: str,
        compute_on_step: bool,
        prefix: str,
        max_gap: int = 5,
        true_doc_path: str = None,
    ):

        self.ent_classes = ent_classes
        self.rel_classes_set = rel_classes_set
        self.ent_id2label = ent_id2label
        self.ent_label2id = {v: k for k, v in self.ent_id2label.items()}
        self.rel_id2label = rel_id2label
        self.rel_label2id = {v: k for k, v in self.rel_id2label.items()}
        self.ent_neg_class = ent_neg_class
        self.rel_neg_class = rel_neg_class
        self.prefix = prefix
        self.max_gap = max_gap

        self.ent_eval = EvalEntWLP(
            classes=ent_classes,
            id2label=ent_id2label,
            neg_class=ent_neg_class,
            prefix=f"{prefix}_ent"
        )

        self.full_eval = EvalRelWLP(
            classes=self.rel_classes_set['full'],
            id2label=rel_id2label,
            neg_class=rel_neg_class,
            prefix=f"{prefix}_FULL",
            keep_groups=FULL,
        )

        self.iap_eval = EvalRelWLP(
            classes=self.rel_classes_set['iap'],
            id2label=rel_id2label,
            neg_class=rel_neg_class,
            prefix=f"{prefix}_IAP",
            keep_groups=[RelationGroup.IAP],
        )

        self.captac_eval = EvalRelWLP(
            classes=self.rel_classes_set["captac"],
            id2label=rel_id2label,
            neg_class=rel_neg_class,
            prefix=f"{prefix}_CAPTAC",
            keep_groups=CAPTAC,
            keep_gaps=list(range(0, max_gap))
        )

        self.intra_captac_eval = EvalRelWLP(
            classes=self.rel_classes_set["captac"],
            id2label=rel_id2label,
            neg_class=rel_neg_class,
            prefix=f"{prefix}_intra_CAPTAC",
            keep_groups=CAPTAC,
            keep_gaps=[0]
        )

        self.inter_captac_eval = EvalRelWLP(
            classes=self.rel_classes_set["captac"],
            id2label=rel_id2label,
            neg_class=rel_neg_class,
            prefix=f"{prefix}_inter_CAPTAC",
            keep_groups=CAPTAC,
            keep_gaps=list(range(1, max_gap))
        )

        self.temp_eval = EvalRelWLP(
            classes=self.rel_classes_set["temp"],
            id2label=rel_id2label,
            neg_class=rel_neg_class,
            prefix=f"{prefix}_TEMP",
            keep_groups=TEMP,
        )

        self.causal_eval = EvalRelWLP(
            classes=self.rel_classes_set["captac"],
            id2label=rel_id2label,
            neg_class=rel_neg_class,
            prefix=f"{prefix}_C",
            keep_groups=[RelationGroup.C]
        )

        self.temp_implicit = {
            RelationGroup(t).name: EvalRelWLP(
                classes=self.rel_classes_set["temp"],
                id2label=rel_id2label,
                neg_class=rel_neg_class,
                prefix=f"{prefix}_TEMP_{RelationGroup(t).name}",
                keep_groups=[t]
            )
            for t in TEMP
        }

        self.captac_dist = {
            i: EvalRelWLP(
                classes=self.rel_classes_set["captac"],
                id2label=rel_id2label,
                neg_class=rel_neg_class,
                prefix=f"{prefix}_CAPTAC_Dist_{i}",
                keep_groups=CAPTAC,
                keep_gaps=[i]
            )
            for i in range(max_gap)
        }

        self.evals = [
            self.full_eval,
            self.iap_eval,
            self.captac_eval,
            self.intra_captac_eval,
            self.inter_captac_eval,
            self.temp_eval,
            self.causal_eval,
        ] + list(self.temp_implicit.values()) + list(self.captac_dist.values())

        with open(true_doc_path, 'r') as t_f:
            self.true_docs = [json.loads(doc) for doc in t_f.readlines()]

        # convert ents and rel dicts to dataframes
        self.true_docs = [
            dict(
                tokens2d=doc['tokens'],
                entities=self.ents_to_df(doc['named_entities']),
                relations=self.rels_to_df(doc['relations']),
                doc_key=doc['doc_key']
            ) for doc in self.true_docs
        ]
        self.pred_doc_merger = DocumentMerger(
            save_dir=self.prefix,
            ent_id2label=ent_id2label,
            rel_id2label=rel_id2label,
            true_doc_tokens={
                doc['doc_key']: doc['tokens2d']
                for doc in self.true_docs
            }
        )

    def ents_to_df(self, ents):
        ents_df = pd.DataFrame(list(it.chain.from_iterable(ents)))
        ents_df['label'] = ents_df.apply(
            lambda ent: self.ent_label2id[ent.label], axis=1
        )
        return ents_df

    def rels_to_df(self, rels):
        return pd.DataFrame(
            [
                {
                    "span1":
                        (
                            rel['arg1_start'],
                            rel['arg1_end'],
                            rel['arg1_step_idx']
                        ),
                    "span2":
                        (
                            rel['arg2_start'],
                            rel['arg2_end'],
                            rel['arg2_step_idx']
                        ),
                    "label": self.rel_label2id[rel['label']]
                } for rel in rels
            ]
        )

    def update(
        self,
        tokens2d: List[List[str]],
        pred_doc: Tuple[pd.DataFrame, pd.DataFrame],
        orig_key: str,
        partition_key: str
    ):
        ents, rels = pred_doc
        self.pred_doc_merger.load_partitions(
            tokens2d=tokens2d,
            ents=ents,
            rels=rels,
            orig_doc_key=orig_key,
            partition_key=int(partition_key)
        )

    def append_relation_group(self, docs):
        return [
            (
                doc_key,
                ents_df,
                RelationGroupFinder(
                    ents_df,
                    rels_df,
                    ent_id2label=self.ent_id2label,
                    rel_id2label=self.rel_id2label
                ).append_rel_group()
            ) for doc_key,
            ents_df,
            rels_df in
            tqdm(docs, desc=f"[{self.prefix}] appending_relation_group")
        ]

    def tabulate_sub_collection(self, dfs, sub_collection, titles):
        dfs = [
            df.reset_index(drop=True) for k,
            df in dfs.items() if k in sub_collection
        ]

        return pd.concat(dfs, keys=titles, axis=1)

    # TODO cleanup
    def tabulate_report(self, tables: Dict[str, Dict[str, pd.DataFrame]]):
        # generate prf for ents
        prf_tables = {
            eval_type: eval_tables['prf']
            for eval_type,
            eval_tables in tables.items()
        }
        per_class_tables = {
            eval_type: eval_tables['per_class']
            for eval_type,
            eval_tables in tables.items()
        }

        prf_mains = [
            f"{self.prefix}_{tag}" for tag in
            ["ent", "FULL", "IAP", "CAPTAC", "intra_CAPTAC", "inter_CAPTAC"]
        ]

        prf_captac = [
            f"{self.prefix}_{tag}" for tag in
            ["TEMP", "TEMP_I_I", "TEMP_I_E", "TEMP_E_I", "TEMP_E_E", "C"]
        ]

        captac_dist = [
            f"{self.prefix}_CAPTAC_Dist_{i}" for i in range(self.max_gap)
        ]

        prf_main_df = self.tabulate_sub_collection(
            prf_tables,
            prf_mains,
            titles=[
                "ent", "FULL", "IAP", "CAPTAC", "intra_CAPTAC", "inter_CAPTAC"
            ]
        )

        prf_captac_df = self.tabulate_sub_collection(
            dfs=prf_tables,
            sub_collection=prf_captac,
            titles=[
                "TEMP", "TEMP_I_I", "TEMP_I_E", "TEMP_E_I", "TEMP_E_E", "C"
            ]
        )

        captac_dist_df = self.tabulate_sub_collection(
            dfs=prf_tables,
            sub_collection=captac_dist,
            titles=[f"Dist_{i}" for i in range(self.max_gap)]
        )

        for k, df in per_class_tables.items():
            df.index.name = k

        return {
            f"{self.prefix}_prf":
                prf_main_df.to_markdown(),
            f"{self.prefix}_prf_captac":
                prf_captac_df.to_markdown(),
            f"{self.prefix}_captac_dist":
                captac_dist_df.to_markdown(),
            f"{self.prefix}_per_class":
                "\n".join([df.to_html() for _, df in per_class_tables.items()])
        }

    def compute(self):
        # merge all document partitions
        pred_docs = self.pred_doc_merger.merge()

        # TEMP ######################
        pred_doc_keys = [doc['doc_key'] for doc in pred_docs]
        true_docs = [
            doc for doc in self.true_docs if doc['doc_key'] in pred_doc_keys
        ]
        # TEMP ##################

        # align pred to true
        doc_keys = [doc['doc_key'] for doc in true_docs]

        pred_docs = [
            (doc['doc_key'], doc['entities'], doc['relations'])
            for doc in pred_docs
        ]
        true_docs = [
            (doc['doc_key'], doc['entities'], doc['relations'])
            for doc in true_docs
        ]

        # append relation group info.
        pred_docs = self.append_relation_group(pred_docs)
        true_docs = self.append_relation_group(true_docs)

        pred_docs = {doc_key: (ents, rels) for doc_key, ents, rels in pred_docs}
        true_docs = {doc_key: (ents, rels) for doc_key, ents, rels in true_docs}

        all_metrics = {}
        all_tables = {}
        all_images = {}

        for doc_key in doc_keys:
            pred_ents, _ = pred_docs[doc_key]
            true_ents, _ = true_docs[doc_key]
            self.ent_eval(pred_ents, true_ents)

        metrics, tables, image = self.ent_eval.compute()
        all_metrics[self.ent_eval.prefix] = metrics
        all_tables[self.ent_eval.prefix] = tables
        all_images[self.ent_eval.prefix] = image

        for eval in tqdm(self.evals):
            for doc_key in doc_keys:
                _, pred_rels = pred_docs[doc_key]
                _, true_rels = true_docs[doc_key]
                # update all evals
                eval(pred_rels, true_rels)

            metrics, tables, image = eval.compute()
            all_metrics[eval.prefix] = metrics
            all_tables[eval.prefix] = tables
            all_images[eval.prefix] = image

        html_tables = self.tabulate_report(all_tables)

        self.reset()

        return sutils.flatten_dict(all_metrics), html_tables, all_images

    def reset(self):
        for obj in self.evals:
            obj.reset()

        self.pred_doc_merger.reset()


if __name__ == "__main__":
    pass
