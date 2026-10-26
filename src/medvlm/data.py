import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

def patient_level_split(
    df,
    train_frac=.70,
    val_frac=.10,
    temp_calib_frac=.05,
    conformal_calib_frac=.05,
    seed=7
):
    """
    Patient-level data split.

    A subject_id can appear in exactly one partition, preventing
    patient leakage across training, validation, calibration,
    conformal calibration, and test sets.

    Partitions:
        train              - model fitting
        val                - checkpoint/threshold selection
        temp_calib         - temperature scaling
        conformal_calib    - conformal calibration
        test               - final evaluation only
    """

    used_frac = (
        train_frac
        + val_frac
        + temp_calib_frac
        + conformal_calib_frac
    )

    if used_frac >= 1:
        raise ValueError(
            "train_frac + val_frac + temp_calib_frac + "
            "conformal_calib_frac must be < 1"
        )

    patients = df["subject_id"].drop_duplicates().to_numpy()

    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n = len(patients)

    a = int(n * train_frac)
    b = a + int(n * val_frac)
    c = b + int(n * temp_calib_frac)
    d = c + int(n * conformal_calib_frac)

    ids = {
        "train": set(patients[:a]),
        "val": set(patients[a:b]),
        "temp_calib": set(patients[b:c]),
        "conformal_calib": set(patients[c:d]),
        "test": set(patients[d:])
    }

    return {
        name: df[df.subject_id.isin(subject_ids)].reset_index(drop=True)
        for name, subject_ids in ids.items()
    }

def encode_labels(row, labels):
    y, mask = [], []
    for name in labels:
        v = row[name]
        # Conservative U-ignore policy: uncertain (-1) and missing targets are excluded.
        if pd.isna(v) or float(v) == -1.0:
            y.append(0.0); mask.append(0.0)
        else:
            y.append(float(v)); mask.append(1.0)
    return torch.tensor(y, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

def clinical_prompt(row):
    parts = []
    if "age" in row and pd.notna(row["age"]):
        parts.append(f"Age: {int(row['age'])}.")
    if "sex" in row and pd.notna(row["sex"]):
        parts.append(f"Sex: {row['sex']}.")
    txt = str(row.get("report_text", "") or "").strip()
    if txt:
        parts.append("Clinical text: " + txt)
    parts.append("Question: Which target chest radiograph findings are present?")
    return " ".join(parts)

class CXRDataset(Dataset):
    def __init__(self, df, labels, tokenizer_name, image_size=224, max_length=128, train=False):
        self.df = df
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self.max_length = max_length
        ops = [transforms.Resize((image_size, image_size))]
        if train:
            # Deliberately conservative; no horizontal flip because laterality can matter.
            ops.append(transforms.RandomAffine(3, translate=(.02,.02), scale=(.98,1.02)))
        ops += [transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                transforms.Normalize([.5,.5,.5],[.5,.5,.5])]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        image = self.transform(Image.open(row.image_path).convert("L"))
        tok = self.tokenizer(
            clinical_prompt(row), truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        y, m = encode_labels(row, self.labels)
        return {
            "image": image,
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "labels": y, "label_mask": m,
            "subject_id": str(row.subject_id),
            "study_id": str(row.study_id),
            "image_path": str(row.image_path)
        }

def loader(df, labels, dcfg, text_backbone, train=False, shuffle=False):
    ds = CXRDataset(df, labels, text_backbone, dcfg.image_size, dcfg.max_text_length, train)
    return DataLoader(ds, batch_size=dcfg.batch_size, shuffle=shuffle,
                      num_workers=dcfg.num_workers, pin_memory=True)
