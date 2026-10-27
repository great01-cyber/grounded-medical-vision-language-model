# MedVLM-Grounded

**Grounded and Trustworthy Medical Vision-Language Learning for Chest X-ray Interpretation**

MedVLM-Grounded is a research framework for investigating **trustworthy multimodal artificial intelligence for chest X-ray interpretation**.

The project studies whether medical vision-language models can combine chest X-ray images with textual or clinical information while providing predictions that are not only accurate, but also **calibrated, uncertainty-aware, spatially grounded, parameter-efficient, and capable of abstaining when confidence is insufficient**.

Rather than treating medical image classification as a prediction-only problem, MedVLM-Grounded investigates the complete pathway from multimodal representation learning to reliability and evidence grounding.

The framework compares three modelling strategies:

1. **Contrastive vision-language learning**
2. **Cross-attention multimodal fusion**
3. **LoRA-adapted cross-attention**

These models are evaluated within a common framework incorporating multilabel classification, temperature scaling, conformal prediction, selective prediction, explanation, and spatial grounding.

> **Status:** Active research project. The current synthetic experiments validate the implementation pipeline only. They are not evidence of clinical performance and should not be interpreted as medical results.

---

# 1. Motivation

Deep learning systems can achieve strong predictive performance in medical imaging, but accuracy alone is insufficient for high-stakes clinical applications.

A model may produce the correct prediction while relying on irrelevant image regions, become overconfident when it encounters unfamiliar cases, or fail without communicating that its prediction is unreliable.

Medical AI therefore raises several questions beyond:

> **Did the model predict the correct label?**

We are also interested in:

> **How confident is the model?**

> **Is that confidence calibrated?**

> **What visual evidence contributed to the prediction?**

> **Does the highlighted evidence correspond to the relevant anatomical region?**

> **Can the model recognise cases where it should not make a confident prediction?**

> **Can multimodal adaptation be achieved without fine-tuning a very large number of parameters?**

MedVLM-Grounded brings these questions into a single experimental framework.

---

# 2. Research Question

The central research question is:

> **Can a medical vision-language model make predictions that are not only accurate, but also calibrated, uncertainty-aware, spatially grounded, and capable of abstaining when confidence is insufficient?**

This is investigated through several sub-questions.

### RQ1 — Multimodal Representation

How effectively can chest X-ray images and clinical text be represented within a common multimodal learning framework?

### RQ2 — Fusion Strategy

Does explicit cross-attention between textual and visual tokens provide advantages over global contrastive vision-language representations?

### RQ3 — Parameter-Efficient Adaptation

Can LoRA-adapted cross-attention retain useful multimodal modelling capacity while reducing the number of trainable parameters?

### RQ4 — Calibration

Are the predicted probabilities well calibrated, and can post-hoc temperature scaling improve their reliability?

### RQ5 — Conformal Prediction

Can conformal prediction provide useful prediction sets under an explicit target error level?

### RQ6 — Selective Prediction

Can uncertainty estimates be used to identify cases where the model should abstain rather than make an unreliable prediction?

### RQ7 — Evidence Grounding

Do the model's visual explanations correspond to clinically relevant image regions rather than merely producing plausible-looking heatmaps?

---

# 3. Target Findings

The initial chest X-ray task considers eight findings:

| Class | Finding |
|---|---|
| 1 | Atelectasis |
| 2 | Cardiomegaly |
| 3 | Consolidation |
| 4 | Edema |
| 5 | Lung Opacity |
| 6 | Pleural Effusion |
| 7 | Pneumonia |
| 8 | Pneumothorax |

The task is **multilabel**, meaning multiple findings may be present simultaneously.

---

# 4. Methodology Overview

The proposed pipeline is:

```text
Chest X-ray ──→ Vision Encoder ───────┐
                                      │
                                      ├──→ Multimodal Model
                                      │          │
Clinical Text ─→ Text Encoder ────────┘          │
                                                 ↓
                                      Multilabel Predictions
                                                 │
                         ┌───────────────────────┼──────────────────────┐
                         ↓                       ↓                      ↓
                    Calibration             Conformal             Explanation
                         │                   Prediction                 │
                         ↓                       │                      ↓
                    Uncertainty                  ↓               Spatial Grounding
                         │                Prediction Sets                │
                         ↓                       │                      ↓
                    Abstention              Coverage              Grounding Metrics
                         └───────────────────────┴──────────────────────┘
                                                 │
                                                 ↓
                                      Trustworthy Prediction
```

The framework intentionally separates **prediction**, **uncertainty**, and **evidence grounding** rather than assuming that a high classification score automatically implies trustworthy behaviour.

---

# 5. Model Architecture

## 5.1 Vision Encoder

The image encoder transforms a chest X-ray

```math
\mathbf{x}\in\mathbb{R}^{H\times W}
```

into a sequence of visual tokens:

```math
V = \{v_1,v_2,\ldots,v_N\}.
```

The current implementation uses a Vision Transformer-style image backbone.

---

## 5.2 Text Encoder

The textual input is tokenised and processed by a biomedical language encoder:

```math
T = \{t_1,t_2,\ldots,t_M\}.
```

The framework currently supports a biomedical BERT-based text backbone.

For downstream diagnostic experiments, textual inputs must be selected carefully to prevent **target leakage**. Reports containing the radiologist's final findings should not simply be supplied as diagnostic input when those same findings constitute the prediction targets.

---

# 6. Multimodal Strategies

## 6.1 Model A — Contrastive Vision-Language Learning

The first model provides a global vision-language baseline.

Image and text representations are projected into a shared embedding space.

Given normalised image and text embeddings

```math
\hat{v}_i
```

and

```math
\hat{t}_j,
```

their similarity is computed as:

```math
s_{ij} = \frac{\hat{v}_i^\top \hat{t}_j}{\tau}
```

where

```math
\tau
```

is a learnable temperature parameter.

A symmetric contrastive objective encourages matching image-text pairs to obtain greater similarity than non-matching pairs.

Conceptually:

```math
\mathcal{L}_{CLIP}
=
\frac{1}{2}
\left(
\mathcal{L}_{image\rightarrow text}
+
\mathcal{L}_{text\rightarrow image}
\right).
```

This provides a baseline for global multimodal alignment.

---

## 6.2 Model B — Cross-Attention Fusion

The second architecture allows textual tokens to directly query visual image tokens.

Cross-attention is defined using:

```math
Q = TW_Q
```

```math
K = VW_K
```

```math
V' = VW_V.
```

Attention is then calculated as:

```math
Attention(Q,K,V')
=
softmax
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V'.
```

This allows textual representations to retrieve information from relevant visual tokens rather than relying only on globally pooled embeddings.

The implementation uses **multi-head scaled dot-product cross-attention**.

---

## 6.3 Model C — LoRA-Adapted Cross-Attention

The third architecture retains the cross-attention structure but introduces **Low-Rank Adaptation (LoRA)** to the query, key, and value projections.

Instead of fully updating a weight matrix

```math
W,
```

LoRA represents the adaptation as:

```math
W' = W + \frac{\alpha}{r}BA
```

where:

```math
A\in\mathbb{R}^{r\times d_{in}}
```

and

```math
B\in\mathbb{R}^{d_{out}\times r}.
```

The rank

```math
r
```

is chosen to be much smaller than the original dimensionality.

In the current implementation, the pretrained encoders and base Q/K/V projections are frozen while the low-rank adapters and other required task-specific components remain trainable.

For the current configuration:

| Model | Total Parameters | Trainable Parameters |
|---|---:|---:|
| Contrastive | 131,595,715 | 300,553 |
| Cross-Attention | 131,790,530 | 495,368 |
| LoRA Cross-Attention | 131,802,818 | 310,280 |

The LoRA cross-attention configuration therefore uses approximately **37.4% fewer trainable parameters** than the full cross-attention configuration.

This comparison concerns **trainable parameters**, not total stored model parameters.

---

# 7. Multilabel Classification

For each sample, the model predicts probabilities for the eight target findings:

```math
\hat{\mathbf{y}}
=
[\hat{y}_1,\hat{y}_2,\ldots,\hat{y}_8].
```

Sigmoid activation converts logits into independent per-class probabilities:

```math
p_k = \sigma(z_k).
```

Training uses masked binary cross-entropy so that missing or explicitly uncertain target labels can be excluded from the corresponding loss terms.

---

# 8. Data Splitting and Leakage Prevention

Medical datasets can contain multiple images or studies from the same patient.

Random image-level splitting can therefore allow information from the same patient to appear in both training and evaluation sets.

MedVLM-Grounded performs splitting at the **patient level**.

The experimental partitions are:

```text
Train
Validation
Temperature-Calibration
Conformal-Calibration
Test
```

A patient can appear in only one partition.

The partitions serve different purposes:

| Partition | Purpose |
|---|---|
| Train | Model parameter optimisation |
| Validation | Model/threshold selection |
| Temperature Calibration | Temperature scaling |
| Conformal Calibration | Conformal threshold estimation |
| Test | Final evaluation only |

Separating the calibration datasets prevents the final test set from being used to fit post-hoc reliability procedures.

---

# 9. Probability Calibration

Neural networks can produce probabilities that do not accurately represent empirical correctness.

MedVLM-Grounded therefore incorporates **temperature scaling**.

For a model logit

```math
z,
```

the calibrated probability is:

```math
p =
\sigma
\left(
\frac{z}{T}
\right)
```

where

```math
T > 0
```

is estimated using held-out calibration data.

Calibration is evaluated using metrics including:

- Expected Calibration Error
- Brier Score
- Reliability analysis

---

# 10. Conformal Prediction

A single probability does not necessarily communicate whether a prediction should be considered sufficiently reliable.

The framework therefore investigates conformal prediction.

Instead of always returning one fixed binary decision, conformal prediction can construct prediction sets such as:

```text
{Positive}
```

```text
{Negative}
```

or, when the evidence is insufficient:

```text
{Positive, Negative}
```

Given an error level

```math
\alpha,
```

the conformal procedure estimates a calibration quantile from held-out data.

Under the assumptions required by the conformal procedure, this provides a **marginal coverage guarantee** rather than a guarantee for every individual patient or subgroup.

---

# 11. Selective Prediction and Abstention

A trustworthy model should not necessarily be forced to predict on every case.

MedVLM-Grounded therefore includes selective prediction.

The model may:

```text
Predict Positive
Predict Negative
Abstain
```

depending on its confidence and uncertainty.

Predictive uncertainty can be measured using binary entropy:

```math
H(p)
=
-p\log(p)
-
(1-p)\log(1-p).
```

Predictions can then be evaluated using **risk-coverage analysis**.

Coverage measures the proportion of cases on which the model makes predictions, while risk measures the corresponding error.

This enables investigation of whether model performance improves when uncertain cases are deferred.

---

# 12. Explainability

Prediction probabilities alone do not reveal which image evidence influenced the model.

The framework therefore includes visual explanation mechanisms that can produce image-level attribution maps.

Candidate approaches include:

- Integrated Gradients
- attention-based visualisation
- gradient-based attribution methods

However:

> **Attention should not automatically be interpreted as explanation, and saliency should not automatically be interpreted as causal evidence.**

Explanation maps are therefore treated as objects that require evaluation rather than as proof that a model reasons clinically.

---

# 13. Spatial Evidence Grounding

The grounding component asks a stronger question:

> **Does the evidence highlighted by the model overlap with the region associated with the target finding?**

Where spatial annotations are available, explanation maps can be compared with reference regions using measures such as:

- Pointing Game
- Energy Inside Ground-Truth Region
- Intersection over Union
- localisation accuracy

MS-CXR is intended to provide spatial annotations for this stage of the project.

The distinction is important:

```text
Classification:
"What pathology does the model predict?"

Explanation:
"What influenced the prediction?"

Grounding:
"Does that evidence correspond to the relevant region in the image?"
```

---

# 14. Evaluation

The final experimental comparison will consider multiple dimensions of model behaviour.

### Predictive Performance

- AUROC
- AUPRC
- F1
- Precision
- Recall

### Calibration

- Expected Calibration Error
- Brier Score
- reliability analysis

### Conformal Prediction

- empirical coverage
- prediction-set size

### Selective Prediction

- risk-coverage curves
- retained coverage
- abstention rate

### Grounding

- Pointing Game
- Energy Inside Ground-Truth Region
- IoU/localisation metrics

### Efficiency

- total parameters
- trainable parameters
- relative reduction in trainable parameters

The objective is therefore not to declare one model superior based on a single AUROC value.

Instead, the models are evaluated across:

```text
Prediction
     +
Calibration
     +
Uncertainty
     +
Selective Prediction
     +
Grounding
     +
Parameter Efficiency
```

---

# 15. Current Experiments

The repository currently contains a synthetic-data pipeline designed to validate the software architecture.

Synthetic experiments verify that the following sequence executes correctly:

```text
Dataset
   ↓
Data Loader
   ↓
Vision + Text Encoders
   ↓
Multimodal Fusion
   ↓
Classification
   ↓
Loss + Backpropagation
   ↓
Checkpoint
   ↓
Temperature Calibration
   ↓
Conformal Calibration
   ↓
Selective Prediction
   ↓
Evaluation
```

These experiments are **engineering smoke tests**.

They should not be interpreted as evidence of medical performance.

---

# 16. Planned Real-Data Experiments

The intended primary dataset is **MIMIC-CXR-JPG**, subject to credentialed access and the applicable Data Use Agreement.

The planned experimental progression is:

```text
MIMIC-CXR-JPG
      ↓
Patient-Level Cohort Construction
      ↓
Image Preprocessing
      ↓
Leakage-Safe Text Construction
      ↓
Contrastive Baseline
      ↓
Cross-Attention
      ↓
LoRA Cross-Attention
      ↓
Calibration
      ↓
Conformal Prediction
      ↓
Selective Prediction
      ↓
Model Comparison
```

Spatial grounding experiments are planned using **MS-CXR** where suitable reference annotations are available.

No protected MIMIC-CXR data will be distributed through this repository.

---

# 17. Preliminary Engineering Results

The synthetic pipeline has successfully trained and evaluated the implemented architectures end-to-end.

For the current model configuration, LoRA reduced the number of trainable parameters in the cross-attention model from:

```text
495,368
```

to:

```text
310,280
```

corresponding to approximately:

```text
37.4%
```

fewer trainable parameters.

Synthetic classification metrics are intentionally not presented as research findings because the synthetic dataset was created for software validation rather than clinical benchmarking.

---

# 18. Repository Structure

```text
medvlm-grounded/
│
├── configs/
│   ├── contrastive.yaml
│   ├── cross_attention.yaml
│   ├── lora.yaml
│   ├── mimic.yaml
│   └── smoke.yaml
│
├── docs/
│   ├── ACADEMIC_METHOD.md
│   └── REFERENCES.md
│
├── scripts/
│   └── make_synthetic.py
│
├── src/
│   └── medvlm/
│       ├── app.py
│       ├── calibration.py
│       ├── config.py
│       ├── conformal.py
│       ├── data.py
│       ├── evaluate.py
│       ├── explain.py
│       ├── grounding.py
│       ├── inference.py
│       ├── losses.py
│       ├── metrics.py
│       ├── models.py
│       ├── selective.py
│       └── train.py
│
├── tests/
│   └── test_core.py
│
├── .gitignore
├── pyproject.toml
└── README.md
```

Dataset files, trained checkpoints, virtual environments, and experiment outputs are excluded from version control.

---

# 19. Reproduction

Clone the repository:

```bash
git clone https://github.com/great01-cyber/grounded-medical-vision-language-model.git
```

Enter the project:

```bash
cd grounded-medical-vision-language-model
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the project:

```bash
python -m pip install -e .
```

Generate the synthetic smoke-test dataset:

```bash
python scripts/make_synthetic.py
```

Run the tests:

```bash
python -m pytest -q
```

Train the smoke-test model:

```bash
python -m medvlm.train --config configs/smoke.yaml
```

Evaluate the model:

```bash
python -m medvlm.evaluate --config configs/smoke.yaml
```

---

# 20. Reproducibility

The experimental framework uses explicit configuration files for model, data, training, and evaluation settings.

The current synthetic configuration includes a fixed random seed:

```text
seed = 7
```

Final real-data experiments should be repeated across multiple random seeds where computationally feasible, with aggregate results reported rather than relying on a single run.

---

# 21. Limitations

Several limitations should be considered when interpreting this project.

### Synthetic results are not clinical results

The current synthetic dataset exists only to verify the implementation.

### Dataset shift

Performance on one hospital dataset does not guarantee equivalent performance across institutions, scanners, patient populations, or acquisition protocols.

### Calibration can deteriorate under distribution shift

A model calibrated on one distribution may become miscalibrated when deployed under another.

### Conformal guarantees depend on assumptions

Conformal prediction provides marginal guarantees under assumptions such as exchangeability. These guarantees should not be interpreted as universal guarantees for every patient or subgroup.

### Explanations are not causal proofs

Attribution maps can identify model sensitivity but do not automatically demonstrate clinical reasoning or causal dependence.

### Text leakage must be controlled

Radiology reports may explicitly contain the pathology being predicted. Downstream multimodal experiments must therefore distinguish legitimate pre-interpretation clinical information from text that directly reveals the target.

### Clinical deployment is outside the current scope

This repository is a research framework and is **not a medical device or clinical decision-support system**.

---

# 22. Intended Contribution

MedVLM-Grounded is intended to provide a reproducible experimental framework for studying the relationship between:

```text
Multimodal Learning
        ↓
Prediction
        ↓
Calibration
        ↓
Uncertainty
        ↓
Abstention
        ↓
Explanation
        ↓
Evidence Grounding
```

The broader goal is to move beyond asking whether a medical AI system can produce the correct answer and toward asking whether its predictions are **reliable, appropriately uncertain, computationally efficient, and supported by verifiable visual evidence**.

---

# 23. Project Status

- [x] Core repository architecture
- [x] Synthetic data generation
- [x] Patient-level splitting
- [x] Contrastive architecture
- [x] Cross-attention architecture
- [x] LoRA cross-attention
- [x] Multilabel training pipeline
- [x] Temperature scaling
- [x] Conformal prediction
- [x] Selective prediction
- [x] Evaluation framework
- [x] Unit tests
- [ ] MIMIC-CXR-JPG data integration
- [ ] Real-data baseline experiments
- [ ] Full three-model comparison
- [ ] Multi-seed experiments
- [ ] MS-CXR grounding evaluation
- [ ] Explanation comparison
- [ ] Final ablation studies
- [ ] Research figures and tables

---

# 24. References

The repository contains a dedicated reference document:

```text
docs/REFERENCES.md
```

The final research experiments will document the literature supporting the vision-language architecture, LoRA adaptation, calibration, conformal prediction, selective classification, explainability, and medical image grounding components.

---

# Disclaimer

**MedVLM-Grounded is intended solely for research and educational purposes.**

It is not intended for diagnosis, treatment, clinical decision-making, or deployment as a medical device.


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
