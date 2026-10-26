import numpy as np

def binary_entropy(p,eps=1e-8):
    p=np.clip(p,eps,1-eps)
    return -(p*np.log(p)+(1-p)*np.log(1-p))

def sample_confidence(probs):
    return np.maximum(probs,1-probs).mean(1)

def sample_entropy(probs):
    return binary_entropy(probs).mean(1)

def abstain_mask(probs,sets=None,confidence_threshold=.8,entropy_threshold=.55):
    a=sample_confidence(probs)<confidence_threshold
    b=sample_entropy(probs)>entropy_threshold
    c=np.zeros(len(probs),dtype=bool)
    if sets is not None:
        c=np.array([any(len(labelset)!=1 for labelset in sample) for sample in sets])
    return a|b|c

def risk_coverage_curve(y,p,mask,points=50):
    conf=sample_confidence(p); order=np.argsort(-conf); rows=[]
    ks=np.unique(np.linspace(1,len(y),min(points,len(y)),dtype=int))
    for k in ks:
        idx=order[:k]; valid=mask[idx].astype(bool)
        pred=(p[idx]>=.5).astype(int)
        risk=float((pred[valid]!=y[idx][valid]).mean()) if valid.any() else float("nan")
        rows.append({"coverage":float(k/len(y)),"hamming_risk":risk})
    return rows
