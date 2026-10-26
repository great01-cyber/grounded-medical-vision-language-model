import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .config import load_config, resolve_device
from .data import patient_level_split, loader
from .models import build_model
from .calibration import PerClassTemperatureScaler
from .conformal import BinaryLabelConformal
from .metrics import multilabel_metrics
from .selective import abstain_mask, risk_coverage_curve


def move(batch, device):
    """Move tensor fields in a batch to the selected device."""
    for key in (
        "image",
        "input_ids",
        "attention_mask",
        "labels",
        "label_mask",
    ):
        batch[key] = batch[key].to(device)

    return batch


@torch.no_grad()
def collect(model, data_loader, device):
    """
    Run inference and collect logits, labels, and label masks.
    """
    logits = []
    labels = []
    masks = []

    model.eval()

    for batch in data_loader:
        batch = move(batch, device)

        output = model(
            batch["image"],
            batch["input_ids"],
            batch["attention_mask"],
        )

        logits.append(output["logits"].cpu())
        labels.append(batch["labels"].cpu())
        masks.append(batch["label_mask"].cpu())

    return (
        torch.cat(logits),
        torch.cat(labels),
        torch.cat(masks),
    )


def main():
    # --------------------------------------------------
    # Command-line arguments
    # --------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    cfg = load_config(args.config)

    device = resolve_device(
        cfg.train.device
    )

    # --------------------------------------------------
    # Load metadata
    # --------------------------------------------------

    df = pd.read_csv(
        cfg.data.csv
    )

    # --------------------------------------------------
    # Patient-level data split
    #
    # Each patient can appear in only one partition.
    # --------------------------------------------------

    splits = patient_level_split(
        df,
        train_frac=cfg.data.train_frac,
        val_frac=cfg.data.val_frac,
        temp_calib_frac=cfg.data.temp_calib_frac,
        conformal_calib_frac=cfg.data.conformal_calib_frac,
        seed=cfg.data.seed,
    )

    # --------------------------------------------------
    # Temperature-calibration loader
    # --------------------------------------------------

    temp_cal_loader = loader(
        splits["temp_calib"],
        cfg.labels,
        cfg.data,
        cfg.model.text_backbone,
    )

    # --------------------------------------------------
    # Conformal-calibration loader
    # --------------------------------------------------

    conformal_cal_loader = loader(
        splits["conformal_calib"],
        cfg.labels,
        cfg.data,
        cfg.model.text_backbone,
    )

    # --------------------------------------------------
    # Test loader
    # --------------------------------------------------

    test_loader = loader(
        splits["test"],
        cfg.labels,
        cfg.data,
        cfg.model.text_backbone,
    )

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------

    model = build_model(
        cfg.model,
        len(cfg.labels),
    ).to(device)

    # --------------------------------------------------
    # Load trained checkpoint
    # --------------------------------------------------

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    # --------------------------------------------------
    # Collect raw model logits
    # --------------------------------------------------

    z_temp, y_temp, m_temp = collect(
        model,
        temp_cal_loader,
        device,
    )

    z_conf, y_conf, m_conf = collect(
        model,
        conformal_cal_loader,
        device,
    )

    z_test, y_test, m_test = collect(
        model,
        test_loader,
        device,
    )

    # ==================================================
    # 1. TEMPERATURE CALIBRATION
    # ==================================================
    #
    # Temperature scaling is fitted ONLY on the
    # temperature-calibration partition.
    #
    # The conformal and test partitions are not used
    # when estimating temperature.
    # --------------------------------------------------

    scaler = PerClassTemperatureScaler(
        len(cfg.labels)
    )

    scaler.fit(
        z_temp,
        y_temp,
        m_temp,
    )

    # --------------------------------------------------
    # Apply temperature scaling to conformal data
    # --------------------------------------------------

    calibrated_conf_logits = scaler.transform(
        z_conf
    )

    p_conf = torch.sigmoid(
        calibrated_conf_logits
    ).numpy()

    # --------------------------------------------------
    # Apply the same fitted temperature scaling to test
    # --------------------------------------------------

    calibrated_test_logits = scaler.transform(
        z_test
    )

    p_test = torch.sigmoid(
        calibrated_test_logits
    ).numpy()

    # --------------------------------------------------
    # Convert labels/masks to NumPy
    # --------------------------------------------------

    y_conf = y_conf.numpy()
    m_conf = m_conf.numpy()

    y_test = y_test.numpy()
    m_test = m_test.numpy()

    # ==================================================
    # 2. CONFORMAL CALIBRATION
    # ==================================================
    #
    # Conformal thresholds are fitted ONLY on the
    # separate conformal-calibration partition.
    #
    # This partition has already been transformed using
    # the temperature scaler fitted above.
    # --------------------------------------------------

    conformal = BinaryLabelConformal(
        cfg.eval.alpha
    )

    conformal.fit(
        p_conf,
        y_conf,
        m_conf,
    )

    # ==================================================
    # 3. TEST PREDICTION
    # ==================================================
    #
    # Test data has not been used for:
    #
    #   - model fitting
    #   - checkpoint selection
    #   - temperature fitting
    #   - conformal calibration
    #
    # It is used only for final evaluation.
    # --------------------------------------------------

    prediction_sets = conformal.predict_sets(
        p_test
    )

    # --------------------------------------------------
    # Selective prediction / abstention
    # --------------------------------------------------

    abstain = abstain_mask(
        p_test,
        prediction_sets,
        cfg.eval.confidence_threshold,
        cfg.eval.entropy_threshold,
    )

    # ==================================================
    # 4. METRICS
    # ==================================================

    per_class_metrics = multilabel_metrics(
        y_test,
        p_test,
        m_test,
        cfg.eval.ece_bins,
    )

    risk_coverage = risk_coverage_curve(
        y_test,
        p_test,
        m_test,
    )

    retained_coverage = float(
        (~abstain).mean()
    )

    # ==================================================
    # 5. REPORT
    # ==================================================

    report = {
        "labels": cfg.labels,

        "temperature":
            scaler.temperature.tolist(),

        "conformal_q":
            conformal.q.tolist(),

        "per_class":
            per_class_metrics,

        "retained_coverage":
            retained_coverage,

        "risk_coverage":
            risk_coverage,

        "method_note": (
            "Temperature scaling is fitted on the "
            "temperature-calibration partition; "
            "conformal thresholds are fitted on a separate "
            "conformal-calibration partition after "
            "temperature scaling; test data is "
            "evaluation-only."
        ),
    }

    # ==================================================
    # 6. SAVE EVALUATION REPORT
    # ==================================================

    output_dir = Path(
        cfg.train.out_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_path = (
        output_dir / "evaluation.json"
    )

    evaluation_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    # ==================================================
    # 7. SAVE CALIBRATION PARAMETERS
    # ==================================================

    calibration_path = (
        output_dir / "calibration.pt"
    )

    torch.save(
        {
            "temperature":
                scaler.temperature.cpu(),

            "conformal_q":
                conformal.q,
        },
        calibration_path,
    )

    # --------------------------------------------------
    # Print final report
    # --------------------------------------------------

    print(
        json.dumps(
            report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()