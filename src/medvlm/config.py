from dataclasses import dataclass, field
from pathlib import Path
import random, yaml, numpy as np, torch

DEFAULT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Lung Opacity", "Pleural Effusion", "Pneumonia", "Pneumothorax"
]

@dataclass
class DataConfig:
    csv: str
    image_size: int = 224
    max_text_length: int = 128
    batch_size: int = 16
    num_workers: int = 4
    train_frac: float = .70
    val_frac: float = .10
    temp_calib_frac: float = .05
    conformal_calib_frac: float = .05
    seed: int = 7

@dataclass
class ModelConfig:
    kind: str = "cross_attention"  # contrastive | cross_attention | lora
    vision_backbone: str = "vit_small_patch16_224"
    text_backbone: str = "microsoft/BiomedVLP-CXR-BERT-specialized"
    embed_dim: int = 256
    num_heads: int = 8
    dropout: float = .1
    freeze_vision: bool = True
    freeze_text: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16

@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    contrastive_weight: float = .2
    grad_clip: float = 1.0
    device: str = "auto"
    out_dir: str = "runs/default"

@dataclass
class EvalConfig:
    alpha: float = .10
    confidence_threshold: float = .80
    entropy_threshold: float = .55
    ece_bins: int = 15

@dataclass
class Config:
    labels: list[str] = field(default_factory=lambda: DEFAULT_LABELS.copy())
    data: DataConfig | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

def load_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    return Config(
        labels=raw.get("labels", DEFAULT_LABELS),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
        eval=EvalConfig(**raw.get("eval", {})),
    )

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
