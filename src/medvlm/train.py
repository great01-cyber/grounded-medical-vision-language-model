import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .config import load_config, seed_everything, resolve_device
from .data import patient_level_split, loader
from .models import build_model
from .losses import masked_bce, symmetric_infonce


def move(b,device):
    for k in ("image","input_ids","attention_mask","labels","label_mask"):
        b[k]=b[k].to(device)
    return b

@torch.no_grad()
def validation_loss(model,dl,device):
    model.eval(); vals=[]
    for b in dl:
        b=move(b,device)
        out=model(b["image"],b["input_ids"],b["attention_mask"])
        vals.append(masked_bce(out["logits"],b["labels"],b["label_mask"]).item())
    return sum(vals)/max(1,len(vals))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); a=ap.parse_args()
    cfg=load_config(a.config); seed_everything(cfg.data.seed); device=resolve_device(cfg.train.device)
    df=pd.read_csv(cfg.data.csv)
    s = patient_level_split(
    df,
    train_frac=cfg.data.train_frac,
    val_frac=cfg.data.val_frac,
    temp_calib_frac=cfg.data.temp_calib_frac,
    conformal_calib_frac=cfg.data.conformal_calib_frac,
    seed=cfg.data.seed
)
    tr=loader(s["train"],cfg.labels,cfg.data,cfg.model.text_backbone,True,True)
    va=loader(s["val"],cfg.labels,cfg.data,cfg.model.text_backbone)
    model=build_model(cfg.model,len(cfg.labels)).to(device)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=cfg.train.lr,weight_decay=cfg.train.weight_decay)
    out=Path(cfg.train.out_dir); out.mkdir(parents=True,exist_ok=True)
    best=float("inf")
    for epoch in range(cfg.train.epochs):
        model.train()
        for b in tqdm(tr,desc=f"epoch {epoch+1}"):
            b=move(b,device)
            o=model(b["image"],b["input_ids"],b["attention_mask"])
            loss=masked_bce(o["logits"],b["labels"],b["label_mask"])
            if cfg.model.kind=="contrastive":
                loss=loss+cfg.train.contrastive_weight*symmetric_infonce(
                    o["image_embedding"],o["text_embedding"],o["logit_scale"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params,cfg.train.grad_clip); opt.step()
        vl=validation_loss(model,va,device)
        print(json.dumps({"epoch":epoch+1,"val_loss":vl}))
        if vl<best:
            best=vl
            torch.save({"model":model.state_dict(),"labels":cfg.labels},out/"best.pt")

if __name__=="__main__": main()