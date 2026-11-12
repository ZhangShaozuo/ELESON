"""F1 with explicit undecided semantics and tie-correct confidence AUROC."""
import numpy as np


def binary_auc(y, score):
    y = np.asarray(y, dtype=bool)
    pos, neg = int(y.sum()), int((~y).sum())
    if not pos or not neg:
        return None
    order = np.argsort(score, kind="stable")
    values = np.asarray(score)[order]
    ranks = np.empty(len(y), dtype=float)
    left = 0
    while left < len(y):
        right = left+1
        while right < len(y) and values[right] == values[left]:
            right += 1
        ranks[order[left:right]] = (left+1+right)/2
        left = right
    return float((ranks[y].sum()-pos*(pos+1)/2)/(pos*neg))


def classification_metrics(y, pred, confidence, threshold=0.):
    y, pred, confidence = np.asarray(y), np.asarray(pred), np.asarray(confidence)
    accepted = confidence > threshold
    confusion = np.zeros((3, 4), dtype=int)
    np.add.at(confusion, (y, np.where(accepted, pred, 3)), 1)
    def per_class(mask):
        results = {}
        for c, name in enumerate(["neither", "elevator", "escalator"]):
            tp = int(((y == c) & (pred == c) & accepted & mask).sum())
            fp = int(((y != c) & (pred == c) & accepted & mask).sum())
            fn = int(((y == c) & ((pred != c) | ~accepted) & mask).sum())
            results[name] = dict(precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1),
                                 f1=2*tp/max(2*tp+fp+fn, 1), support=int(((y == c)&mask).sum()))
        return results
    all_rows = per_class(np.ones(len(y), dtype=bool))
    covered = per_class(accepted)
    return dict(n=len(y), threshold=threshold, coverage=float(accepted.mean()) if len(y) else 0.,
                ud_ratio=float((~accepted).mean()) if len(y) else 0.,
                accuracy=float((y == pred).mean()) if len(y) else 0.,
                selective_accuracy=float((y[accepted] == pred[accepted]).mean()) if accepted.any() else None,
                conveyor_f1=float(np.mean([all_rows[k]["f1"] for k in ("elevator", "escalator")])),
                selective_conveyor_f1=float(np.mean([covered[k]["f1"] for k in ("elevator", "escalator")])) if accepted.any() else None,
                confidence_auroc=binary_auc(y == pred, confidence),
                per_class=all_rows, selective_per_class=covered, confusion_3x4=confusion.tolist())
