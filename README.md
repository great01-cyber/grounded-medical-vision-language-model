# MedVLM-Grounded

**Research software only — not a medical device and not for clinical use.**

A complete research scaffold for a multimodal chest-X-ray system that consumes:

`image + patient context + clinical question`

and produces:

`multilabel prediction + calibrated uncertainty + conformal label sets + abstention + image attribution + localization grounding`

## The research contribution

This is deliberately more than “fine-tune a VLM.” It runs a controlled comparison of:

1. **CLIP-style global contrastive learning** using symmetric InfoNCE;
2. **cross-attention multimodal fusion**;
3. **LoRA parameter-efficient adaptation** of the cross-modal projections.

The second research axis tests whether a model that predicts well also **looks in the right place**. Integrated Gradients maps are evaluated against radiologist-annotated MS-CXR boxes using pointing-game accuracy, attribution energy inside the box, and IoU.

The third axis treats uncertainty as an empirical research question. The pipeline fits held-out temperature scaling, builds split-conformal binary label sets, and evaluates an abstention rule with a risk–coverage curve.

## Data

For the real study use:
- MIMIC-CXR / MIMIC-CXR-JPG for radiographs, reports and structured labels.
- MS-CXR v1.1 for radiologist phrase/bounding-box grounding evaluation.

Expected CSV:

```text
image_path,subject_id,study_id,report_text,sex,age,Atelectasis,Cardiomegaly,Consolidation,Edema,Lung Opacity,Pleural Effusion,Pneumonia,Pneumothorax
```

MIMIC is credentialed data. Do not commit it, patient identifiers, reports, or derived private files to GitHub.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

## Smoke test

```bash
python scripts/make_synthetic.py --out data/synthetic
pytest -q
python -m medvlm.train --config configs/smoke.yaml
python -m medvlm.evaluate --config configs/smoke.yaml --checkpoint runs/smoke/best.pt
```

The synthetic dataset only tests software plumbing. It has no scientific value.

## Three-model experiment

```bash
python -m medvlm.train --config configs/contrastive.yaml
python -m medvlm.train --config configs/cross_attention.yaml
python -m medvlm.train --config configs/lora.yaml
```

For the real dataset, copy/edit `configs/mimic.yaml`.

## Correct experimental order

```text
patient-level dataset
   ├─ train       -> optimize weights
   ├─ validation  -> model selection / operating choices
   ├─ calibration -> temperature + conformal quantiles
   └─ test        -> untouched final metrics

image ── vision encoder ── image tokens ─┐
                                         ├─ fusion ─ classifier ─ calibrated p
text + patient context ─ text encoder ───┘                    │
                                                              ├─ conformal set
                                                              ├─ abstention
                                                              └─ IG attribution
                                                                    │
                                                              MS-CXR boxes
                                                                    │
                                                        grounding metrics
```

## Important caveats

- Patient-level splitting is mandatory to avoid patient leakage.
- MIMIC uncertain/missing targets are masked by default.
- Calibration and conformal fitting never use test labels.
- Split conformal provides **marginal coverage under exchangeability**, not universal per-patient certainty.
- Attention is not automatically an explanation.
- A heatmap is not “clinically grounded” until evaluated against expert localization.
- The repository does **not** output hidden chain-of-thought. “Explanation” means a concise evidence summary grounded in predictions, uncertainty estimates and attribution measurements.
- The abstention sentence is a research-system behavior: **“ABSTAIN — insufficient confidence; refer to clinician.”**

See `docs/ACADEMIC_METHOD.md` and `docs/REFERENCES.md`.


## Interactive research interface

After training and evaluation, launch the Gradio dashboard:

```bash
python -m medvlm.evaluate \
  --config configs/cross_attention.yaml \
  --checkpoint runs/cross_attention/best.pt

python -m medvlm.app \
  --config configs/cross_attention.yaml \
  --checkpoint runs/cross_attention/best.pt \
  --calibration runs/cross_attention/calibration.pt
```

Open `http://127.0.0.1:7860`.

The dashboard accepts a chest X-ray, age, sex, clinical information and a clinical question. It displays calibrated per-finding probabilities, conformal label sets, uncertainty, the abstention decision and an Integrated Gradients overlay.

For comparisons among CLIP, cross-attention and LoRA, train/evaluate each configuration and launch the corresponding checkpoint. Keep the same patient split/seed for a fair experiment.

**Do not use the public `--share` option with credentialed MIMIC data.** Keep protected clinical data local and comply with the relevant data-use agreement.
