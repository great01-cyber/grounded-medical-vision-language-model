from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from .calibration import PerClassTemperatureScaler
from .config import load_config, resolve_device
from .conformal import BinaryLabelConformal
from .explain import integrated_gradients_image
from .models import build_model
from .selective import abstain_mask, sample_confidence, sample_entropy


class ResearchUI:
    def __init__(self, config_path: str, checkpoint_path: str,
                 calibration_path: str | None = None):
        self.cfg = load_config(config_path)
        self.device = resolve_device(self.cfg.train.device)
        self.model = build_model(self.cfg.model, len(self.cfg.labels)).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model.text_backbone, trust_remote_code=True
        )
        self.transform = transforms.Compose([
            transforms.Resize((self.cfg.data.image_size, self.cfg.data.image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
        ])

        self.scaler = None
        self.conformal = None
        if calibration_path and Path(calibration_path).exists():
            payload = torch.load(calibration_path, map_location="cpu")
            if "temperature" in payload:
                self.scaler = PerClassTemperatureScaler(len(self.cfg.labels))
                self.scaler.log_t = torch.as_tensor(payload["temperature"]).float().log()
            if "conformal_q" in payload:
                self.conformal = BinaryLabelConformal(self.cfg.eval.alpha)
                self.conformal.q = np.asarray(payload["conformal_q"], dtype=float)

    def _prompt(self, age, sex, patient_info, question):
        parts = []
        if age is not None:
            parts.append(f"Age: {int(age)}.")
        if sex:
            parts.append(f"Sex: {sex}.")
        if patient_info and patient_info.strip():
            parts.append("Clinical information: " + patient_info.strip())
        parts.append("Question: " + (question.strip() if question else
                     "Which target chest radiograph findings are present?"))
        return " ".join(parts)

    def _batch(self, image, age, sex, patient_info, question):
        pil = image.convert("L") if isinstance(image, Image.Image) else Image.fromarray(image).convert("L")
        x = self.transform(pil).unsqueeze(0).to(self.device)
        tok = self.tokenizer(
            self._prompt(age, sex, patient_info, question),
            truncation=True, padding="max_length",
            max_length=self.cfg.data.max_text_length, return_tensors="pt"
        )
        return {
            "image": x,
            "input_ids": tok["input_ids"].to(self.device),
            "attention_mask": tok["attention_mask"].to(self.device),
        }, pil

    @staticmethod
    def _overlay(pil, heat):
        base = np.asarray(pil.resize((heat.shape[1], heat.shape[0]))).astype(float)
        base = np.stack([base, base, base], axis=-1)
        # Simple transparent red-yellow style overlay without extra plotting dependencies.
        h = np.clip(heat, 0, 1)
        color = np.zeros_like(base)
        color[..., 0] = 255 * h
        color[..., 1] = 170 * np.sqrt(h)
        mixed = np.clip(.72 * base + .55 * color, 0, 255).astype("uint8")
        return Image.fromarray(mixed)

    def analyse(self, image, age, sex, patient_info, question, target_label):
        if image is None:
            raise gr.Error("Upload a chest X-ray first.")

        batch, original = self._batch(image, age, sex, patient_info, question)
        with torch.no_grad():
            logits = self.model(
                batch["image"], batch["input_ids"], batch["attention_mask"]
            )["logits"]

        calibrated_logits = self.scaler.transform(logits) if self.scaler else logits
        probs = torch.sigmoid(calibrated_logits).cpu().numpy()[0]

        conformal_sets = None
        if self.conformal is not None:
            conformal_sets = self.conformal.predict_sets(probs[None, :])[0]

        abst = bool(abstain_mask(
            probs[None, :],
            [conformal_sets] if conformal_sets is not None else None,
            self.cfg.eval.confidence_threshold,
            self.cfg.eval.entropy_threshold
        )[0])

        if target_label not in self.cfg.labels:
            target_label = self.cfg.labels[int(np.argmax(probs))]
        c = self.cfg.labels.index(target_label)

        # IG uses the diagnostic logit. This is an attribution, not a causal explanation.
        heat = integrated_gradients_image(self.model, batch, c, n_steps=32)
        overlay = self._overlay(original, heat)

        table = pd.DataFrame({
            "Finding": self.cfg.labels,
            "Probability": np.round(probs, 4),
            "Prediction": np.where(probs >= .5, "Present", "Absent"),
            "Conformal set": [
                str(s) if conformal_sets is not None else "not loaded"
                for s in (conformal_sets if conformal_sets is not None else self.cfg.labels)
            ],
        }).sort_values("Probability", ascending=False)

        confidence = float(sample_confidence(probs[None, :])[0])
        entropy = float(sample_entropy(probs[None, :])[0])
        top = int(np.argmax(probs))
        decision = (
            "### ⚠️ ABSTAIN\n**The model is insufficiently confident. Refer to clinician.**"
            if abst else
            f"### Research model output\nMost probable target finding: **{self.cfg.labels[top]}** "
            f"({probs[top]:.1%})."
        )
        uncertainty = (
            f"**Mean binary confidence:** {confidence:.3f}  \n"
            f"**Mean predictive entropy:** {entropy:.3f}  \n"
            f"**Temperature calibration:** {'loaded' if self.scaler else 'not loaded'}  \n"
            f"**Conformal prediction:** {'loaded' if self.conformal else 'not loaded'}"
        )
        explanation = (
            f"Integrated Gradients was computed for **{target_label}**. "
            "The heatmap shows pixels with larger absolute attribution to that diagnostic logit. "
            "It should not be interpreted as causal clinical reasoning. For the research study, "
            "localization must be scored against expert MS-CXR bounding boxes."
        )
        return table, overlay, decision, uncertainty, explanation


def build_demo(engine: ResearchUI):
    with gr.Blocks(title="MedVLM-Grounded Research Interface") as demo:
        gr.Markdown(
            "# MedVLM-Grounded\n"
            "**Multimodal Medical Vision–Language Research Interface**  \n"
            "⚠️ Research software only. Not a medical device and not for clinical decision-making."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Chest X-ray")
                age = gr.Number(value=60, minimum=0, maximum=120, label="Age")
                sex = gr.Dropdown(["F", "M", "Other / unspecified"], label="Sex")
                patient_info = gr.Textbox(lines=4, label="Patient / clinical information",
                                          placeholder="e.g. fever, cough, oxygen saturation...")
                question = gr.Textbox(
                    value="Which target chest radiograph findings are present?",
                    label="Clinical question"
                )
                target = gr.Dropdown(engine.cfg.labels, value=engine.cfg.labels[0],
                                     label="Attribution target")
                run = gr.Button("Analyse case", variant="primary")
            with gr.Column(scale=2):
                result = gr.Dataframe(label="Calibrated model assessment", interactive=False)
                decision = gr.Markdown()
                uncertainty = gr.Markdown()
        with gr.Row():
            overlay = gr.Image(label="Integrated Gradients evidence map")
            explanation = gr.Markdown()

        gr.Markdown(
            "### Interpretation boundary\n"
            "The interface exposes measured predictions, uncertainty and attribution. "
            "It does not expose or claim hidden chain-of-thought. A visually plausible "
            "attribution map is not considered clinically grounded until it is evaluated "
            "against expert localization annotations."
        )

        run.click(
            engine.analyse,
            inputs=[image, age, sex, patient_info, question, target],
            outputs=[result, overlay, decision, uncertainty, explanation]
        )
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--calibration", default=None,
                    help="Optional .pt containing temperature and conformal_q.")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    engine = ResearchUI(args.config, args.checkpoint, args.calibration)
    build_demo(engine).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
