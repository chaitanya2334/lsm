import itertools as it
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.metrics import Metric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src import utils as sutils
from src.postprocessing.tensorboard import plot_confusion_matrix
from src.pytorch_models import utils as putils
from torch import Tensor
from torch.nn import functional as F

###  EvalRE(pred, true, top_span_mask)


class EvalRE(Metric):
    def __init__(
        self,
        classes_set: Dict[str, List[str]],
        id2label: Dict[int, str],
        neg_class: str,
        compute_on_step: bool,
        prefix: str
    ):
        """
        Used to evaluate relations. The result is broken down into:
        1. Full Relation Eval
        2. IAP
        3. CAP - Intra (or within) Sentence
        4. CAP - Inter (or cross) Sentence

        It can be used to evaluate relation predictions based on gold or 
        non-gold entities by correctly passing the `pred_span_mask` in 
        `update()` function. Ths `pred_span_masks` must be of size 
        (batch_size, max_num_spans) where each row represents all spans for 
        that sentence. 

        Args:
            full_classes (List[str]): [description]
            iap_classes (List[str]): [description]
            cap_classes (List[str]): [description]
            id2label (Dict[int, str]): [description]
            neg_class (str): [description]
            compute_on_step (bool): [description]
            prefix (str): [description]
        """
        super().__init__(compute_on_step=compute_on_step)

        self.neg_class = neg_class
        self.classes = classes_set
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.prefix = prefix
        self.neg_class_id = self.label2id[self.neg_class]
        self.max_step_gap = 5
        self.print_errors = False

        states = [
            'full_preds',
            'full_target',
            # 'iap_preds',
            # 'iap_target',
            # 'cap_intra_preds',
            # 'cap_intra_target',
            # 'cap_inter_preds',
            # 'cap_inter_target'
        ]

        for state in states:
            self.add_state(state, default=[])

    def predict(self, rels, ents_df, cross_sentence=True):
        # TODO optimize
        if rels.shape[0] == 0:
            # no predictions to be made
            pred_df = pd.DataFrame(
                {
                    'span1': [], 'span2': [], 'pred_label': []
                },
                columns=['span1', 'span2', 'pred_label']
            )
            return pred_df

        if cross_sentence:
            pred_spans = ents_df.values
            span1, span2 = zip(*it.product(pred_spans, repeat=2))
        else:
            spans_by_sents = ents_df.groupby('step_idx')
            span_pairs = spans_by_sents.apply(
                lambda x: list(it.product(x.values, repeat=2))
            ).values.tolist()
            # remove self loops
            span_pairs = [
                (s1, s2)
                for s1,
                s2 in it.chain.from_iterable(span_pairs)
                if not np.array_equal(s1, s2)
            ]
            if len(span_pairs) > 0:
                span1, span2 = zip(*span_pairs)
            else:
                span1 = []
                span2 = []

        span1 = np.array(span1)
        span2 = np.array(span2)
        # to list of tuples

        if span1.size != 0 and span2.size != 0:
            span1 = list(zip(span1[:, 0], span1[:, 1], span1[:, 2]))
            span2 = list(zip(span2[:, 0], span2[:, 1], span2[:, 2]))
        else:
            span1 = []
            span2 = []

        pred_rels = F.softmax(rels, dim=-1)
        pred_rels = torch.argmax(pred_rels, dim=-1)
        pred_rels = pred_rels.view(-1).cpu().detach().numpy()
        pred_df = pd.DataFrame(
            {
                'span1': span1, 'span2': span2, 'pred_label': pred_rels
            },
            columns=['span1', 'span2', 'pred_label']
        )

        # dont store negative samples (memory and time complexity optimization)
        pred_df = pred_df[pred_df.pred_label != self.neg_class_id]

        return pred_df

    def update(self, pred_df: Tensor, true_df: Tensor):
        """
        Update preds and target lists for relation evaluation based on spans to 
        keep in `pred_span_mask`.

        Args:
            pred (Tensor): 3d tensor of size (max_n_spans, max_n_spans, nb_rels)
                containing predicted relation unnormalized scores. 
            true (Tensor): 2d tensor of size (max_n_spans, max_n_spans) 
                containing ground truth relation label ids for each span pair.
            pred_span_mask (Tensor): 2d tensor of size (batch_size, num_spans) 
                containing boolean values of whether to keep or ignore spans in
                each sentence in the batch. 
            spans (Tensor): 3d tensor of size (batch_size, num_spans, 2) the 
                exact spans defined by their start and end indices. Useful to
                figure out if a relation is IAP, CAP-inter, or CAP-intra.

        """

        merged = pred_df.set_index(
            ['span1', 'span2']
        ).join(true_df.set_index(['span1', 'span2']), how='outer')

        merged = merged.fillna(self.neg_class_id)

        pred_labels = merged.pred_label.values.astype(int)
        true_labels = merged.true_label.values.astype(int)
        # pred_rels -> (max_n_spans, max_n_spans)

        self.full_preds.append(pred_labels)
        self.full_target.append(true_labels)

        if self.print_errors:
            print(merged[merged.pred_label != merged.true_label])

        # TODO implement
        # # eval iap
        # self.iap_preds.append(self.make_iap(pred_rels, spans))
        # self.iap_target.append(self.make_iap(true, spans))

        # # eval csr intra
        # self.csr_intra_preds.append(self.make_csr_intra(pred_rels, spans))
        # self.csr_intra_target.append(self.make_csr_intra(true, spans))

        # # eval csr inter
        # self.csr_inter_preds.append(self.make_csr_inter(pred_rels, spans))
        # self.csr_inter_target.append(self.make_csr_inter(true, spans))

    def compute_by_type(self, preds, target, eval_type):
        preds = np.concatenate(preds)
        target = np.concatenate(target)

        preds = [self.id2label[t] for t in preds.tolist()]
        target = [self.id2label[t] for t in target.tolist()]
        p, r, f1, _ = precision_recall_fscore_support(
            y_pred=preds,
            y_true=target,
            average='micro',
            labels=self.classes[eval_type]
        )
        acc = accuracy_score(y_pred=preds, y_true=target)
        p_class, r_class, f1_class, _ = precision_recall_fscore_support(
            y_true=target,
            y_pred=preds,
            average=None,
            labels=self.classes[eval_type]
        )
        image = plot_confusion_matrix(
            true=target,
            pred=preds,
            classes=self.classes[eval_type] + [self.neg_class],
            title='Confusion matrix for Relations'
        )

        return sutils.flatten_dict(
                {
                    f'{self.prefix}_{eval_type}_rel':
                        {
                            f'{self.prefix}_{eval_type}_rel_acc': acc,
                            f'{self.prefix}_{eval_type}_rel_p': p,
                            f'{self.prefix}_{eval_type}_rel_r': r,
                            f'{self.prefix}_{eval_type}_rel_f1': f1
                        },
                    f"{self.prefix}_{eval_type}_rel_per_class":
                        {
                            label: {
                                "P": p_class[i],
                                "R": r_class[i],
                                "F1": f1_class[i],
                            }
                            for i,
                            label in enumerate(self.classes[eval_type])
                        }
                }
            ), image

    def compute(self):
        coll = [
            ("full", self.full_preds, self.full_target),
            # ("iap", self.iap_preds, self.iap_target),
            # ("cap_inter", self.cap_inter_preds, self.cap_inter_target),
            # ("cap_intra", self.cap_intra_preds, self.cap_intra_target)
        ]
        ret = dict()
        for eval_type, preds, target in coll:
            ret[eval_type] = self.compute_by_type(preds, target, eval_type)

        return ret
