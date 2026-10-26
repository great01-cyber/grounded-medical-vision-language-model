import numpy as np

def bbox_mask(shape,boxes):
    h,w=shape; m=np.zeros((h,w),bool)
    for x,y,bw,bh in boxes:
        x0,y0=max(0,int(x)),max(0,int(y))
        x1,y1=min(w,int(np.ceil(x+bw))),min(h,int(np.ceil(y+bh)))
        m[y0:y1,x0:x1]=True
    return m

def pointing_game(heat,boxes):
    gt=bbox_mask(heat.shape,boxes)
    y,x=np.unravel_index(np.argmax(heat),heat.shape)
    return float(gt[y,x])

def energy_inside_box(heat,boxes):
    gt=bbox_mask(heat.shape,boxes)
    pos=np.clip(heat,0,None)
    return float(pos[gt].sum()/(pos.sum()+1e-8))

def threshold_iou(heat,boxes,quantile=.85):
    gt=bbox_mask(heat.shape,boxes)
    pred=heat>=np.quantile(heat,quantile)
    return float(np.logical_and(gt,pred).sum()/(np.logical_or(gt,pred).sum()+1e-8))
