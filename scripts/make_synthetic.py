from pathlib import Path
import argparse, numpy as np, pandas as pd
from PIL import Image,ImageDraw

LABELS=["Atelectasis","Cardiomegaly","Consolidation","Edema","Lung Opacity","Pleural Effusion","Pneumonia","Pneumothorax"]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="data/synthetic"); ap.add_argument("--patients",type=int,default=80); a=ap.parse_args()
    out=Path(a.out); (out/"images").mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(7); rows=[]
    for sid in range(a.patients):
        for st in range(2):
            arr=np.clip(rng.normal(100,25,(224,224)),0,255).astype("uint8")
            img=Image.fromarray(arr); d=ImageDraw.Draw(img); ys=rng.binomial(1,.2,len(LABELS))
            for c,v in enumerate(ys):
                if v:
                    x=10+(c%4)*50; y=20+(c//4)*90; d.ellipse((x,y,x+30,y+30),fill=190)
            path=out/"images"/f"{sid}_{st}.png"; img.save(path)
            row={"image_path":str(path),"subject_id":sid,"study_id":f"{sid}-{st}",
                 "report_text":"Synthetic smoke-test record; not clinical data.","sex":"F" if sid%2 else "M","age":40+sid%40}
            row.update(dict(zip(LABELS,ys))); rows.append(row)
    pd.DataFrame(rows).to_csv(out/"metadata.csv",index=False)
    print(out/"metadata.csv")
if __name__=="__main__": main()
