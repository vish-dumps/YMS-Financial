"""Assignment score and diagnostics for complete prediction vectors."""

from math import sqrt

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def final_score(precision, recall):
    if precision <= 0 or recall <= 0:
        return 0.0
    return sqrt(1 / (0.75 / precision + 0.25 / recall))


def classification(y, predicted):
    y, predicted = np.asarray(y, dtype=bool), np.asarray(predicted, dtype=bool)
    tp = int(np.sum(y & predicted))
    fp = int(np.sum(~y & predicted))
    fn = int(np.sum(y & ~predicted))
    tn = int(np.sum(~y & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=precision, recall=recall, score=final_score(precision, recall),
                tp=tp, fp=fp, fn=fn, tn=tn, fraction=float(predicted.mean()) if len(y) else 0.0,
                rows=len(y), passed=precision > 0.60)


def diagnostics(y, probability, predicted):
    result = classification(y, predicted)
    if np.unique(y).size == 2:
        result.update(pr_auc=float(average_precision_score(y, probability)),
                      roc_auc=float(roc_auc_score(y, probability)))
    return result
