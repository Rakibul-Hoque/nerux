from __future__ import annotations
from ..tensor import Tensor
from ..tensor import functional as F
from ..tensor import factory as init
import numpy as np


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def _array(value, *, name="input"):
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, Tensor):
        arr = value.data
    if arr.dtype == object:
        raise TypeError(f"{name} must contain numeric or label data, not objects.")
    return arr


def _same_length(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same number of samples; "
            f"got {len(y_true)} and {len(y_pred)}."
        )


def _class_labels(y):
    """Convert one-hot targets to class IDs; otherwise flatten single-column data."""
    y = _array(y, name="y")
    if y.ndim == 2 and y.shape[1] > 1:
        return np.argmax(y, axis=1)
    return y.reshape(-1)


def _score_vector(y_score):
    scores = _array(y_score, name="y_score")
    if scores.ndim == 2:
        if scores.shape[1] == 1:
            scores = scores[:, 0]
        elif scores.shape[1] == 2:
            # Common binary convention: second column is positive-class score.
            scores = scores[:, 1]
        else:
            raise ValueError("Expected binary scores with shape (n,) or (n, 2).")
    if scores.ndim != 1:
        raise ValueError("Expected a one-dimensional score array.")
    return scores.astype(float, copy=False)


def _binary_labels(y_true):
    y = _class_labels(y_true)
    values = np.unique(y)
    if not np.all(np.isin(values, [0, 1, False, True])):
        raise ValueError("Binary metrics require y_true labels encoded as 0 and 1.")
    return y.astype(int, copy=False)


def _binary_predictions(y_pred, threshold=0.5):
    pred = _array(y_pred, name="y_pred")
    if pred.ndim == 2 and pred.shape[1] == 2:
        pred = pred[:, 1]
    elif pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if pred.ndim == 1 and np.all(np.isin(np.unique(pred), [0, 1, False, True])):
        return pred.astype(int, copy=False)
    if pred.ndim != 1:
        raise ValueError("Binary predictions must have shape (n,) or (n, 1)/(n, 2).")
    return (pred >= threshold).astype(int)


def _average(values, supports, average):
    values = np.asarray(values, dtype=float)
    supports = np.asarray(supports, dtype=float)
    if average == "macro":
        return float(np.mean(values)) if values.size else 0.0
    if average == "weighted":
        return float(np.average(values, weights=supports)) if supports.sum() else 0.0
    if average == "micro":
        raise ValueError("Micro averaging must be computed from global TP/FP/FN.")
    if average is None:
        return values
    raise ValueError("average must be 'binary', 'micro', 'macro', 'weighted', or None.")


# Regression metrics


def mean_absolute_error(y_true, y_pred):
    """Mean absolute error (MAE). Supports multi-output arrays."""
    yt, yp = _array(y_true, name="y_true"), _array(y_pred, name="y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"Shapes must match; got {yt.shape} and {yp.shape}.")
    return float(np.mean(np.abs(yp - yt)))


def mean_squared_error(y_true, y_pred):
    """Mean squared error (MSE). Supports multi-output arrays."""
    yt, yp = _array(y_true, name="y_true"), _array(y_pred, name="y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"Shapes must match; got {yt.shape} and {yp.shape}.")
    return float(np.mean((yp - yt) ** 2))


def root_mean_squared_error(y_true, y_pred):
    """Root mean squared error (RMSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def median_absolute_error(y_true, y_pred):
    """Median absolute error."""
    yt, yp = _array(y_true, name="y_true"), _array(y_pred, name="y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"Shapes must match; got {yt.shape} and {yp.shape}.")
    return float(np.median(np.abs(yp - yt)))


def max_error(y_true, y_pred):
    """Maximum absolute prediction error."""
    yt, yp = _array(y_true, name="y_true"), _array(y_pred, name="y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"Shapes must match; got {yt.shape} and {yp.shape}.")
    return float(np.max(np.abs(yp - yt)))


def mean_absolute_percentage_error(y_true, y_pred, *, zero="ignore"):
    """
    Mean absolute percentage error, returned as a ratio (0.1 = 10%).

    zero='ignore' skips zero targets; zero='raise' rejects them.
    """
    yt, yp = (
        _array(y_true, name="y_true").astype(float),
        _array(y_pred, name="y_pred").astype(float),
    )
    if yt.shape != yp.shape:
        raise ValueError(f"Shapes must match; got {yt.shape} and {yp.shape}.")
    mask = yt != 0
    if zero == "raise" and not np.all(mask):
        raise ValueError("MAPE is undefined for zero-valued targets.")
    if zero not in ("ignore", "raise"):
        raise ValueError("zero must be 'ignore' or 'raise'.")
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])))


def r2_score(y_true, y_pred):
    """Coefficient of determination (R²). Returns 0.0 if targets are constant."""
    yt, yp = (
        _array(y_true, name="y_true").astype(float),
        _array(y_pred, name="y_pred").astype(float),
    )
    if yt.shape != yp.shape:
        raise ValueError(f"Shapes must match; got {yt.shape} and {yp.shape}.")
    residual_sum = float(np.sum((yt - yp) ** 2))
    total_sum = float(np.sum((yt - np.mean(yt)) ** 2))
    if total_sum == 0:
        return 1.0 if residual_sum == 0 else 0.0
    return float(1.0 - residual_sum / total_sum)


def explained_variance_score(y_true, y_pred):
    """Explained variance score."""
    yt, yp = (
        _array(y_true, name="y_true").astype(float),
        _array(y_pred, name="y_pred").astype(float),
    )
    if yt.shape != yp.shape:
        raise ValueError(f"Shapes must match; got {yt.shape} and {yp.shape}.")
    target_var = float(np.var(yt))
    if target_var == 0:
        return 1.0 if np.allclose(yt, yp) else 0.0
    return float(1.0 - np.var(yt - yp) / target_var)


# Basic classification metrics


def accuracy_score(y_true, y_pred, *, threshold=0.5):
    """Accuracy for binary/multiclass class IDs or one-hot multiclass arrays."""
    yt = _class_labels(y_true)
    yp_raw = _array(y_pred, name="y_pred")
    if yp_raw.ndim == 2 and yp_raw.shape[1] > 1:
        yp = np.argmax(yp_raw, axis=1)
    elif yp_raw.ndim == 2 and yp_raw.shape[1] == 1:
        yp = _binary_predictions(yp_raw, threshold)
    elif yp_raw.ndim == 1:
        if np.issubdtype(yp_raw.dtype, np.floating) and not np.all(
            np.isin(np.unique(yp_raw), [0.0, 1.0])
        ):
            yp = (yp_raw >= threshold).astype(int)
        else:
            yp = yp_raw
    else:
        raise ValueError("Unsupported y_pred shape.")
    _same_length(yt, yp)
    return float(np.mean(yt == yp))


def confusion_matrix(y_true, y_pred, *, labels=None):
    """Confusion matrix: rows are true labels, columns are predicted labels."""
    yt, yp = _class_labels(y_true), _class_labels(y_pred)
    _same_length(yt, yp)
    if labels is None:
        labels = np.unique(np.concatenate([yt, yp]))
    labels = np.asarray(labels)
    lookup = {label: i for i, label in enumerate(labels.tolist())}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true_label, pred_label in zip(yt, yp):
        if true_label in lookup and pred_label in lookup:
            cm[lookup[true_label], lookup[pred_label]] += 1
    return cm


def precision_score(
    y_true, y_pred, *, average="binary", pos_label=1, threshold=0.5, zero_division=0
):
    """Precision. average: binary, micro, macro, weighted, or None."""
    yt = _class_labels(y_true)
    yp_raw = _array(y_pred, name="y_pred")
    if yp_raw.ndim == 2 and yp_raw.shape[1] > 1:
        yp = np.argmax(yp_raw, axis=1)
    elif average == "binary":
        yp = _binary_predictions(yp_raw, threshold)
    else:
        yp = _class_labels(yp_raw)
    _same_length(yt, yp)

    if average == "binary":
        tp = np.sum((yt == pos_label) & (yp == pos_label))
        fp = np.sum((yt != pos_label) & (yp == pos_label))
        denom = tp + fp
        return float(tp / denom) if denom else float(zero_division)

    labels = np.unique(np.concatenate([yt, yp]))
    cm = confusion_matrix(yt, yp, labels=labels)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp

    supports = cm.sum(axis=1)
    per_class = np.divide(
        tp, tp + fp, out=np.full_like(tp, float(zero_division)), where=(tp + fp) != 0
    )
    if average == "micro":
        denom = tp.sum() + fp.sum()
        return float(tp.sum() / denom) if denom else float(zero_division)
    return _average(per_class, supports, average)


def recall_score(
    y_true, y_pred, *, average="binary", pos_label=1, threshold=0.5, zero_division=0
):
    """Recall. average: binary, micro, macro, weighted, or None."""
    yt = _class_labels(y_true)
    yp_raw = _array(y_pred, name="y_pred")
    if yp_raw.ndim == 2 and yp_raw.shape[1] > 1:
        yp = np.argmax(yp_raw, axis=1)
    elif average == "binary":
        yp = _binary_predictions(yp_raw, threshold)
    else:
        yp = _class_labels(yp_raw)
    _same_length(yt, yp)

    if average == "binary":
        tp = np.sum((yt == pos_label) & (yp == pos_label))
        fn = np.sum((yt == pos_label) & (yp != pos_label))
        denom = tp + fn
        return float(tp / denom) if denom else float(zero_division)

    labels = np.unique(np.concatenate([yt, yp]))
    cm = confusion_matrix(yt, yp, labels=labels)
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - tp
    supports = cm.sum(axis=1)
    per_class = np.divide(
        tp, tp + fn, out=np.full_like(tp, float(zero_division)), where=(tp + fn) != 0
    )
    if average == "micro":
        denom = tp.sum() + fn.sum()
        return float(tp.sum() / denom) if denom else float(zero_division)
    return _average(per_class, supports, average)


def f1_score(
    y_true, y_pred, *, average="binary", pos_label=1, threshold=0.5, zero_division=0
):
    """F1 score. average: binary, micro, macro, weighted, or None."""
    p = precision_score(
        y_true,
        y_pred,
        average=average,
        pos_label=pos_label,
        threshold=threshold,
        zero_division=zero_division,
    )
    r = recall_score(
        y_true,
        y_pred,
        average=average,
        pos_label=pos_label,
        threshold=threshold,
        zero_division=zero_division,
    )
    if average is None:
        p, r = np.asarray(p), np.asarray(r)
        return np.divide(
            2 * p * r, p + r, out=np.zeros_like(p, dtype=float), where=(p + r) != 0
        )
    return float(2 * p * r / (p + r)) if (p + r) else float(zero_division)


def specificity_score(y_true, y_pred, *, pos_label=1, threshold=0.5, zero_division=0):
    """Binary specificity (true-negative rate)."""
    yt = _binary_labels(y_true)
    yp = _binary_predictions(y_pred, threshold)
    _same_length(yt, yp)
    tn = np.sum((yt != pos_label) & (yp != pos_label))
    fp = np.sum((yt != pos_label) & (yp == pos_label))
    return float(tn / (tn + fp)) if (tn + fp) else float(zero_division)


def balanced_accuracy_score(y_true, y_pred, *, threshold=0.5):
    """Mean recall across classes; for binary classification this is balanced accuracy."""
    yt = _class_labels(y_true)
    yp_raw = _array(y_pred, name="y_pred")
    yp = (
        np.argmax(yp_raw, axis=1)
        if yp_raw.ndim == 2 and yp_raw.shape[1] > 1
        else _binary_predictions(yp_raw, threshold)
    )
    _same_length(yt, yp)
    labels = np.unique(yt)
    recalls = [np.mean(yp[yt == label] == label) for label in labels]
    return float(np.mean(recalls)) if recalls else 0.0


def matthews_corrcoef(y_true, y_pred, *, threshold=0.5):
    """Matthews correlation coefficient (MCC), binary or multiclass."""
    yt = _class_labels(y_true)
    yp_raw = _array(y_pred, name="y_pred")
    yp = (
        np.argmax(yp_raw, axis=1)
        if yp_raw.ndim == 2 and yp_raw.shape[1] > 1
        else _binary_predictions(yp_raw, threshold)
    )
    _same_length(yt, yp)
    cm = confusion_matrix(yt, yp)
    t_sum = cm.sum(axis=1).astype(float)
    p_sum = cm.sum(axis=0).astype(float)
    correct = float(np.trace(cm))
    total = float(cm.sum())
    numerator = correct * total - float(np.dot(t_sum, p_sum))
    denominator = np.sqrt(
        (total**2 - np.dot(p_sum, p_sum)) * (total**2 - np.dot(t_sum, t_sum))
    )
    return float(numerator / denominator) if denominator else 0.0


def top_k_accuracy_score(y_true, y_score, *, k=5):
    """Top-k accuracy for multiclass score/probability matrices."""
    yt = _class_labels(y_true)
    scores = _array(y_score, name="y_score")
    if scores.ndim != 2:
        raise ValueError("y_score must have shape (n_samples, n_classes).")
    _same_length(yt, scores)
    if not 1 <= k <= scores.shape[1]:
        raise ValueError(f"k must be between 1 and {scores.shape[1]}.")
    topk = np.argpartition(scores, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(topk == yt[:, None], axis=1)))


def roc_auc_score(y_true, y_score):
    """Binary ROC-AUC computed from scores using the trapezoidal ROC curve."""
    yt = _binary_labels(y_true)
    scores = _score_vector(y_score)
    _same_length(yt, scores)
    positives = int(np.sum(yt == 1))
    negatives = int(np.sum(yt == 0))
    if positives == 0 or negatives == 0:
        raise ValueError(
            "ROC-AUC requires at least one positive and one negative sample."
        )

    order = np.argsort(scores, kind="mergesort")
    yt_sorted = yt[order]
    score_sorted = scores[order]
    # Rank-sum formula handles tied scores by assigning average ranks.
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and score_sorted[j] == score_sorted[i]:
            j += 1
        ranks[i:j] = ((i + 1) + j) / 2.0  # 1-based average rank
        i = j
    positive_rank_sum = float(np.sum(ranks[yt_sorted == 1]))
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    )


# Multilabel metrics


def hamming_loss(y_true, y_pred, *, threshold=0.5):
    """Fraction of individual labels predicted incorrectly (multilabel)."""
    yt = _array(y_true, name="y_true").astype(int)
    yp = _array(y_pred, name="y_pred")
    if yp.shape != yt.shape:
        raise ValueError(f"Shapes must match; got {yt.shape} and {yp.shape}.")
    if yp.dtype.kind in "fc":
        yp = (yp >= threshold).astype(int)
    else:
        yp = yp.astype(int)
    return float(np.mean(yt != yp))


def exact_match_score(y_true, y_pred, *, threshold=0.5):
    """Multilabel subset accuracy: every label in a sample must match."""
    yt = _array(y_true, name="y_true").astype(int)
    yp = _array(y_pred, name="y_pred")
    if yp.shape != yt.shape:
        raise ValueError(f"Shapes must match; got {yt.shape} and {yp.shape}.")
    if yp.dtype.kind in "fc":
        yp = (yp >= threshold).astype(int)
    else:
        yp = yp.astype(int)
    return float(np.mean(np.all(yt == yp, axis=1)))


def multilabel_f1_score(
    y_true, y_pred, *, average="micro", threshold=0.5, zero_division=0
):
    """F1 for multilabel indicator matrices; average='micro', 'macro', or 'samples'."""
    yt = _array(y_true, name="y_true").astype(int)
    yp = _array(y_pred, name="y_pred")
    if yp.shape != yt.shape or yt.ndim != 2:
        raise ValueError("y_true and y_pred must be matching 2D multilabel matrices.")
    yp = (yp >= threshold).astype(int) if yp.dtype.kind in "fc" else yp.astype(int)

    if average == "samples":
        vals = []
        for t, p in zip(yt, yp):
            tp = np.sum((t == 1) & (p == 1))
            fp = np.sum((t == 0) & (p == 1))
            fn = np.sum((t == 1) & (p == 0))
            denom = 2 * tp + fp + fn
            vals.append(2 * tp / denom if denom else zero_division)
        return float(np.mean(vals))

    tp = np.sum((yt == 1) & (yp == 1), axis=0).astype(float)
    fp = np.sum((yt == 0) & (yp == 1), axis=0).astype(float)
    fn = np.sum((yt == 1) & (yp == 0), axis=0).astype(float)
    if average == "micro":
        denom = 2 * tp.sum() + fp.sum() + fn.sum()
        return float(2 * tp.sum() / denom) if denom else float(zero_division)
    per_label = np.divide(
        2 * tp,
        2 * tp + fp + fn,
        out=np.full_like(tp, float(zero_division)),
        where=(2 * tp + fp + fn) != 0,
    )
    if average == "macro":
        return float(np.mean(per_label))
    if average is None:
        return per_label
    raise ValueError("average must be 'micro', 'macro', 'samples', or None.")


__all__ = [
    "accuracy_score",
    "balanced_accuracy_score",
    "confusion_matrix",
    "exact_match_score",
    "explained_variance_score",
    "f1_score",
    "hamming_loss",
    "max_error",
    "matthews_corrcoef",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "median_absolute_error",
    "multilabel_f1_score",
    "precision_score",
    "r2_score",
    "recall_score",
    "roc_auc_score",
    "root_mean_squared_error",
    "specificity_score",
    "top_k_accuracy_score",
]
