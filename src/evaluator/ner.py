import itertools as it
from collections import defaultdict
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from pytorch_lightning.metrics import Metric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src import utils as sutils
from src.evaluator.f1 import per_class_scores
from src.postprocessing.tensorboard import plot_confusion_matrix
from src.pytorch_models import utils as putils


class EvalNER(Metric):
    def __init__(
        self, classes, id2label, neg_class, compute_on_step=True, prefix=""
    ):
        """
        Meta NER evaluator
        """
        super().__init__(compute_on_step=compute_on_step)
        self.classes = classes
        self.id2label = id2label
        self.prefix = prefix
        self.neg_class = neg_class
        self.label2id = {v: k for k, v in id2label.items()}
        self.neg_class_id = self.label2id[self.neg_class]

        self.add_state("preds", default=[])
        self.add_state("target", default=[])

    def predict(self, ent_logits, spans, span_mask):
        pred_spans = putils.tensor_flatten_spans(spans, span_mask)
        # -> (valid_spans, 3)

        softmax = F.softmax(ent_logits, dim=1)
        labels = torch.argmax(softmax, dim=1)
        # -> (valid_spans, 1)

        out = torch.cat([pred_spans, labels.unsqueeze(1)], dim=1)
        # -> (valid_spans, 4)

        return pd.DataFrame(
            out.cpu().detach().numpy(),
            columns=["start", "end", "step_idx", "label"]
        )

    def update(self, pred_df: pd.DataFrame, true_df: pd.DataFrame):

        merged = pred_df.set_index(['start', 'end', 'step_idx']).join(
            true_df.set_index(['start', 'end', 'step_idx']),
            how='outer',
            lsuffix="_pred",
            rsuffix="_true"
        )

        merged = merged.fillna(self.neg_class_id)

        pred_labels = merged.label_pred.values.astype(int)
        true_labels = merged.label_true.values.astype(int)

        self.preds.append(pred_labels)
        self.target.append(true_labels)

    def plot_cm(self, preds, target):

        image = plot_confusion_matrix(
            true=target,
            pred=preds,
            classes=self.classes + [self.neg_class],
            title='Confusion matrix for Entities'
        )

        return image

    def compute(self):
        preds = np.concatenate(self.preds)
        target = np.concatenate(self.target)

        preds = [self.id2label[t] for t in preds.tolist()]
        target = [self.id2label[t] for t in target.tolist()]

        acc = accuracy_score(y_pred=preds, y_true=target)

        p, r, f1, s = precision_recall_fscore_support(
            y_true=target,
            y_pred=preds,
            labels=self.classes,
            average='micro'
        )

        p_class, r_class, f1_class, s_class = precision_recall_fscore_support(
            y_true=target,
            y_pred=preds,
            labels=self.classes
        )

        image = self.plot_cm(preds, target)

        return sutils.flatten_dict(
            {
                f"{self.prefix}_ent_acc": acc,
                f"{self.prefix}_ent_micro": {
                    "P": p,
                    "R": r,
                    "F1": f1
                },
                f"{self.prefix}_ent_per_class": {
                    label: {
                        "P": p_class[i],
                        "R": r_class[i],
                        "F1": f1_class[i],
                    } for i, label in enumerate(self.classes) if s_class[i] != 0
                }
            }
        ), image
