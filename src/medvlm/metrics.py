import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
)


def ece_binary(y, p, n_bins=15):
    """
    Expected Calibration Error for binary predictions.
    """
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0

    for i, (lo, hi) in enumerate(
        zip(edges[:-1], edges[1:])
    ):
        if i == 0:
            idx = (p >= lo) & (p <= hi)
        else:
            idx = (p > lo) & (p <= hi)

        if idx.any():
            out += (
                idx.mean()
                * abs(
                    y[idx].mean()
                    - p[idx].mean()
                )
            )

    return float(out)


def select_f1_thresholds(
    y,
    p,
    mask,
    default_threshold=0.5,
):
    """
    Select one decision threshold per class using validation data.

    The threshold that maximizes F1 on the validation partition
    is selected independently for each class.

    Parameters
    ----------
    y:
        Ground-truth binary labels with shape [N, C].

    p:
        Predicted probabilities with shape [N, C].

    mask:
        Label-validity mask with shape [N, C].

    default_threshold:
        Threshold used when a class cannot be optimized because
        there are insufficient valid labels.

    Returns
    -------
    np.ndarray
        One threshold per class.
    """
    n_classes = y.shape[1]

    thresholds = np.full(
        n_classes,
        default_threshold,
        dtype=np.float32,
    )

    for c in range(n_classes):
        idx = mask[:, c].astype(bool)

        yc = y[idx, c]
        pc = p[idx, c]

        # We cannot meaningfully optimize F1 if there are no
        # observations or only one target class is present.
        if (
            len(yc) == 0
            or len(np.unique(yc)) < 2
        ):
            continue

        # Candidate thresholds are derived from the actual
        # validation probabilities.
        candidates = np.unique(
            np.concatenate(
                (
                    np.array([0.0]),
                    pc,
                    np.array([1.0]),
                )
            )
        )

        best_threshold = default_threshold
        best_f1 = -1.0

        for threshold in candidates:
            pred = (
                pc >= threshold
            ).astype(int)

            score = f1_score(
                yc,
                pred,
                zero_division=0,
            )

            if score > best_f1:
                best_f1 = score
                best_threshold = threshold

        thresholds[c] = float(
            best_threshold
        )

    return thresholds


def multilabel_metrics(
    y,
    p,
    mask,
    n_bins=15,
    thresholds=None,
):
    """
    Compute per-class multilabel metrics.

    AUROC, AUPRC, Brier score, and ECE use probabilities
    directly.

    F1 uses the supplied class-specific decision thresholds.
    If thresholds are not supplied, 0.5 is used for backwards
    compatibility.
    """
    n_classes = y.shape[1]

    if thresholds is None:
        thresholds = np.full(
            n_classes,
            0.5,
            dtype=np.float32,
        )

    thresholds = np.asarray(
        thresholds,
        dtype=np.float32,
    )

    if len(thresholds) != n_classes:
        raise ValueError(
            "Number of thresholds must match "
            "number of classes."
        )

    rows = []

    for c in range(n_classes):
        idx = mask[:, c].astype(bool)

        yc = y[idx, c]
        pc = p[idx, c]

        threshold = float(
            thresholds[c]
        )

        if (
            len(yc) == 0
            or len(np.unique(yc)) < 2
        ):
            rows.append(
                {
                    "auroc": None,
                    "auprc": None,
                    "f1": None,
                    "brier": None,
                    "ece": None,
                    "threshold": threshold,
                    "n": int(len(yc)),
                }
            )
            continue

        pred = (
            pc >= threshold
        ).astype(int)

        rows.append(
            {
                "auroc": float(
                    roc_auc_score(
                        yc,
                        pc,
                    )
                ),

                "auprc": float(
                    average_precision_score(
                        yc,
                        pc,
                    )
                ),

                "f1": float(
                    f1_score(
                        yc,
                        pred,
                        zero_division=0,
                    )
                ),

                "brier": float(
                    np.mean(
                        (pc - yc) ** 2
                    )
                ),

                "ece": ece_binary(
                    yc,
                    pc,
                    n_bins,
                ),

                "threshold":
                    threshold,

                "n":
                    int(len(yc)),
            }
        )

    return rows