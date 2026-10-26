# Academic method and safeguards

## Research question
Compare global contrastive alignment, explicit cross-attention fusion, and parameter-efficient LoRA adaptation under the same patient-level split. Then ask whether improvements in discrimination also improve calibration, selective reliability, and spatial explanation grounding.

## Data design
Use MIMIC-CXR/MIMIC-CXR-JPG for paired radiographs, reports and labels. Use MS-CXR for a separate localization/phrase-grounding analysis. Split by patient, not image. Never tune on the test partition.

## Targets
The default eight findings match the MS-CXR grounding categories closely enough to support a coherent classification-to-localization study. MIMIC-style uncertain (-1) and missing labels are masked (U-ignore) rather than silently treated as negative.

## Models
1. Contrastive baseline: symmetric InfoNCE between global image/text embeddings plus a multilabel classifier.
2. Cross-attention: clinical-text tokens query image tokens; the pooled fused representation predicts findings.
3. LoRA: the same fusion architecture, but low-rank trainable updates are inserted into frozen cross-modal projections.

The code deliberately does not call attention weights an explanation.

## Explainability
Integrated Gradients is applied to the selected class logit. Evaluate the resulting map against radiologist boxes using:
- pointing-game hit;
- fraction of positive attribution energy inside boxes;
- thresholded IoU.

A qualitative heatmap alone is not evidence of faithfulness.

## Calibration
Fit temperature parameters on the calibration split only. Report Brier score and ECE alongside AUROC/AUPRC. AUROC measures ranking/discrimination, not probability calibration.

## Conformal uncertainty
For each finding, use split-conformal binary label sets with nonconformity `1 - p(y|x)` and the finite-sample corrected quantile. The guarantee is marginal under exchangeability; it is not a guarantee under arbitrary domain shift or for every subgroup.

## Abstention
Abstain when calibrated confidence is low, entropy is high, or a conformal label set is ambiguous. Report risk versus retained coverage. Select thresholds on validation/calibration data, never by optimizing test performance.

## Clinical language
Do not expose chain-of-thought as “clinical reasoning.” The safe research output is a concise evidence summary: predicted finding, calibrated probability, uncertainty/conformal set, and spatial attribution/grounding score. This avoids implying that generated prose faithfully reveals the model's internal causal reasoning.

## Statistical protocol for a serious paper
Use multiple patient-level random seeds. Report bootstrap 95% confidence intervals for AUROC/AUPRC and paired comparisons on the same test cases. Pre-specify the primary metric and the abstention operating point. Add subgroup analyses only where sample size and labels support them. For external validity, evaluate on an external CXR dataset/hospital if licensing permits.
