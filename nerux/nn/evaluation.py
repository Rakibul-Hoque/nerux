import numpy as np
from ..tensor import functional as F
from ..data.dataloader import DataLoader


def evaluation(model, dataset, batch_size=128, threshold=0.5, top_k=5):
    loader = DataLoader(dataset, batch_size, shuffle=False)
    predictions = []
    targets = []

    for xb, yb in loader:
        pred = model(xb)
        predictions.append(pred)
        targets.append(yb)

    y_pred = F.concat(predictions, axis=0).data
    y_true = F.concat(targets, axis=0).data

    task = _detect_task(y_true)

    if task == "binary_classification":
        return _eval_binary(y_pred, y_true, threshold)
    elif task == "multiclass_classification":
        return _eval_multiclass(y_pred, y_true, top_k)
    elif task == "multilabel_classification":
        return _eval_multilabel(y_pred, y_true, threshold)
    else:
        return _eval_regression(y_pred, y_true)


def _detect_task(y_true):
    if y_true.ndim == 1 or y_true.shape[1] == 1:
        unique_vals = np.unique(y_true)
        if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
            return "binary_classification"
        return "regression"
    else:
        if np.all((y_true == 0) | (y_true == 1)):
            if np.all(np.sum(y_true, axis=1) == 1):
                return "multiclass_classification"
            return "multilabel_classification"
        return "regression"


def _eval_binary(y_pred, y_true, threshold):
    y_pred_class = (y_pred >= threshold).astype(int).flatten()
    y_true_class = y_true.flatten().astype(int)
    n = len(y_true_class)

    tp = int(np.sum((y_pred_class == 1) & (y_true_class == 1)))
    tn = int(np.sum((y_pred_class == 0) & (y_true_class == 0)))
    fp = int(np.sum((y_pred_class == 1) & (y_true_class == 0)))
    fn = int(np.sum((y_pred_class == 0) & (y_true_class == 1)))

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0

    # ROC-AUC via trapezoidal rule
    roc_auc = _roc_auc(y_pred.flatten(), y_true_class)

    return {
        "task": "binary_classification",
        "samples": n,
        "accuracy": round(float(accuracy), 4),
        "balanced_accuracy": round(float(balanced_accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "npv": round(float(npv), 4),
        "f1_score": round(float(f1), 4),
        "mcc": round(float(mcc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],  # [[TN, FP], [FN, TP]]
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }


def _eval_multiclass(y_pred, y_true, top_k=5):
    y_pred_class = np.argmax(y_pred, axis=1)
    y_true_class = np.argmax(y_true, axis=1)
    total = len(y_true_class)
    classes = np.arange(y_true.shape[1])
    n_classes = len(classes)

    accuracy = float(np.sum(y_pred_class == y_true_class) / total)

    # Top-k accuracy
    actual_k = min(top_k, n_classes)
    top_k_preds = np.argsort(y_pred, axis=1)[:, -actual_k:]
    top_k_acc = float(
        np.mean([y_true_class[i] in top_k_preds[i] for i in range(total)])
    )

    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true_class, y_pred_class):
        cm[t][p] += 1

    # Per-class metrics
    per_class = {}
    precisions, recalls, f1s, supports = [], [], [], []

    for cls in classes:
        tp = int(cm[cls, cls])
        fp = int(np.sum(cm[:, cls]) - tp)
        fn = int(np.sum(cm[cls, :]) - tp)
        support = int(np.sum(y_true_class == cls))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

        per_class[int(cls)] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    supports_arr = np.array(supports)
    weighted_f1 = float(np.average(f1s, weights=supports_arr))
    weighted_prec = float(np.average(precisions, weights=supports_arr))
    weighted_rec = float(np.average(recalls, weights=supports_arr))

    return {
        "task": "multiclass_classification",
        "samples": total,
        "num_classes": n_classes,
        "accuracy": round(accuracy, 4),
        f"top_{actual_k}_accuracy": round(top_k_acc, 4),
        "macro_precision": round(float(np.mean(precisions)), 4),
        "macro_recall": round(float(np.mean(recalls)), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "weighted_precision": round(weighted_prec, 4),
        "weighted_recall": round(weighted_rec, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def _eval_multilabel(y_pred, y_true, threshold):
    y_pred_bin = (y_pred >= threshold).astype(int)
    y_true_bin = y_true.astype(int)
    n_samples, n_labels = y_true_bin.shape

    exact_match = float(np.mean(np.all(y_pred_bin == y_true_bin, axis=1)))
    hamming_accuracy = float(np.sum(y_pred_bin == y_true_bin) / (n_samples * n_labels))

    # Label cardinality / density
    label_cardinality = float(np.mean(np.sum(y_true_bin, axis=1)))
    label_density = label_cardinality / n_labels

    per_label = {}
    precisions, recalls, f1s = [], [], []
    micro_tp = micro_fp = micro_fn = 0

    for i in range(n_labels):
        tp = int(np.sum((y_pred_bin[:, i] == 1) & (y_true_bin[:, i] == 1)))
        fp = int(np.sum((y_pred_bin[:, i] == 1) & (y_true_bin[:, i] == 0)))
        fn = int(np.sum((y_pred_bin[:, i] == 0) & (y_true_bin[:, i] == 1)))
        support = int(np.sum(y_true_bin[:, i]))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        per_label[i] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    micro_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_rec = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_prec * micro_rec / (micro_prec + micro_rec)
        if (micro_prec + micro_rec) > 0
        else 0.0
    )

    return {
        "task": "multilabel_classification",
        "samples": n_samples,
        "num_labels": n_labels,
        "label_cardinality": round(label_cardinality, 4),
        "label_density": round(label_density, 4),
        "exact_match": round(exact_match, 4),
        "hamming_accuracy": round(hamming_accuracy, 4),
        "micro_precision": round(micro_prec, 4),
        "micro_recall": round(micro_rec, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(float(np.mean(precisions)), 4),
        "macro_recall": round(float(np.mean(recalls)), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "per_label": per_label,
    }


def _eval_regression(y_pred, y_true):
    y_pred_f = y_pred.flatten()
    y_true_f = y_true.flatten()
    n = len(y_true_f)

    residuals = y_pred_f - y_true_f
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    medae = float(np.median(np.abs(residuals)))
    max_err = float(np.max(np.abs(residuals)))

    # MAPE — guard against zero targets
    nonzero = y_true_f != 0
    mape = (
        float(np.mean(np.abs(residuals[nonzero] / y_true_f[nonzero])) * 100)
        if nonzero.any()
        else None
    )

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true_f - np.mean(y_true_f)) ** 2))
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    # Explained variance
    exp_var = 1 - float(np.var(residuals)) / (float(np.var(y_true_f)) + 1e-10)

    return {
        "task": "regression",
        "samples": n,
        "mse": round(mse, 6),
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "median_ae": round(medae, 6),
        "max_error": round(max_err, 6),
        "mape_%": round(mape, 4) if mape is not None else None,
        "r2": round(r2, 4),
        "explained_variance": round(exp_var, 4),
    }


def _roc_auc(y_scores, y_true):
    """Trapezoidal ROC-AUC — no sklearn needed."""
    thresholds = np.sort(np.unique(y_scores))[::-1]
    tprs, fprs = [0.0], [0.0]
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return 0.0
    for t in thresholds:
        pred = (y_scores >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        tprs.append(tp / pos)
        fprs.append(fp / neg)
    tprs.append(1.0)
    fprs.append(1.0)
    return float(np.trapz(tprs, fprs))
