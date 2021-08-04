from collections import defaultdict

from src.corpus.seqeval import (
    f1_score,
    precision_score,
    recall_score,
)
from src.corpus.seqeval import get_entities
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np


def count_rels(links, rel_type):
    return len([link for link in links if link.l_name == rel_type])


def count_correct(preds, trues, rel_type):
    def correct(p_link, t_link, r_type):
        return p_link.arg1 == t_link.arg1 and p_link.arg2 == t_link.arg2 \
            and (p_link.l_name == t_link.l_name == r_type)

    count = 0
    for pred_link in preds:
        for true_link in trues:
            if correct(pred_link, true_link, rel_type):
                count += 1

    return count


def precision(pred_links, true_links, rel_type):
    n_correct = count_correct(pred_links, true_links, rel_type)
    n_extracted = count_rels(pred_links, rel_type)
    if n_extracted == 0:
        return 1

    return n_correct / n_extracted


def recall(pred_links, true_links, rel_type):
    n_correct = count_correct(pred_links, true_links, rel_type)
    n_gold = count_rels(true_links, rel_type)
    if n_gold == 0:
        return 1

    return n_correct / n_gold


# TODO tests
def macro_scores(pred_links, true_links, rels):
    p = []
    r = []
    for i, rel in enumerate(rels):
        p.append(precision(pred_links, true_links, rel))
        r.append(recall(pred_links, true_links, rel))

    print("precision: {}".format(p))
    print("recall: {}".format(r))
    p_mean = sum(p) / len(rels)
    r_mean = sum(r) / len(rels)

    f1 = (2 * p_mean * r_mean) / (p_mean + r_mean)

    return p_mean, r_mean, f1


def per_class_scores(y_pred, y_true, classes):
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))

    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))

    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))

    p_class, r_class, f1_class, s_class = {}, {}, {}, {}

    # if we dont have any entry in pred or true for a specific class,
    # set it to zeros.
    for c in classes:
        p_class[c] = 0
        r_class[c] = 0
        f1_class[c] = 0
        s_class[c] = 0

    for type_name, true_entities in d1.items():
        pred_entities = d2[type_name]
        nb_correct = len(true_entities & pred_entities)
        nb_pred = len(pred_entities)
        nb_true = len(true_entities)

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        p_class[type_name] = p
        r_class[type_name] = r
        f1_class[type_name] = f1
        s_class[type_name] = nb_true

    return p_class, r_class, f1_class, s_class


def report(y_pred, y_true, classes, prefix="", eval_type="sklearn"):
    micro_p, micro_r, micro_f1, _ = prfs(
        y_true=y_true,
        y_pred=y_pred,
        average='micro',
        labels=classes,
        eval_type=eval_type
    )

    macro_p, macro_r, macro_f1, _ = prfs(
        y_true=y_true,
        y_pred=y_pred,
        average='macro',
        labels=classes,
        eval_type=eval_type
    )

    p_class, r_class, f_class, s_class = prfs(
        y_true=y_true,
        y_pred=y_pred,
        labels=classes,
        eval_type=eval_type
    )

    return {
        f"{prefix}_acc": accuracy_score(y_true=y_true, y_pred=y_pred),
        f"{prefix}_micro": {
            "P": micro_p, "R": micro_r, "F1": micro_f1
        },
        f"{prefix}_macro": {
            "P": macro_p, "R": macro_r, "F1": macro_f1
        },
        f"{prefix}_per_class":
            {
                k: {
                    "P": p_class[i],
                    "R": r_class[i],
                    "F1": f_class[i],
                }
                for i,
                k in enumerate(classes)
                if s_class[i] != 0
            }
    }


def prfs(y_pred, y_true, average=None, labels=None, eval_type="sklearn"):
    if eval_type == "sklearn":
        return precision_recall_fscore_support(
            y_true=y_true, y_pred=y_pred, average=average, labels=labels
        )

    elif eval_type == "seqeval":
        return seqeval_prfs(
            y_true=y_true, y_pred=y_pred, average=average, labels=labels
        )


def seqeval_prfs(y_true, y_pred, average, labels):
    if average == "micro":
        p = precision_score(y_true=y_true, y_pred=y_pred)
        r = recall_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)
        s = None

    elif average == "macro":
        p_class, r_class, f_class, s_class = per_class_scores(y_true=y_true,
                                                              y_pred=y_pred,
                                                              classes=labels)

        ps = list(p_class.values())
        rs = list(r_class.values())
        f1s = list(f_class.values())
        p = np.average(ps)
        r = np.average(rs)
        f1 = np.average(f1s)
        s = len(get_entities(y_true))
    elif average is None:
        p, r, f1, s = per_class_scores(y_true=y_true,
                                       y_pred=y_pred,
                                       classes=labels)

        # convert to sklearn interface
        p = np.array([p[c] for c in labels])
        r = np.array([r[c] for c in labels])
        f1 = np.array([f1[c] for c in labels])
        s = np.array([s[c] for c in labels])

    else:
        raise ValueError("Unsupported average {}".format(average))

    return p, r, f1, s
