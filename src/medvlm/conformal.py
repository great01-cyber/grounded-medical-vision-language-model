import math, numpy as np

def corrected_quantile(scores,alpha):
    scores=np.asarray(scores,float)
    if len(scores)==0: return np.nan
    level=min(1.0,math.ceil((len(scores)+1)*(1-alpha))/len(scores))
    return float(np.quantile(scores,level,method="higher"))

class BinaryLabelConformal:
    """
    Per-finding split-conformal label sets using score 1-p(y|x).
    The usual guarantee is marginal under exchangeability, not conditional
    coverage for every patient subgroup and not arbitrary distribution shift.
    """
    def __init__(self,alpha=.1):
        self.alpha=alpha; self.q=None

    def fit(self,probs,y,mask):
        qs=[]
        for c in range(probs.shape[1]):
            idx=mask[:,c].astype(bool)
            pc=probs[idx,c]; yc=y[idx,c].astype(int)
            ptrue=np.where(yc==1,pc,1-pc)
            qs.append(corrected_quantile(1-ptrue,self.alpha))
        self.q=np.asarray(qs)
        return self

    def predict_sets(self,probs):
        result=[]
        for row in probs:
            sample=[]
            for c,p1 in enumerate(row):
                labels=[]
                if p1 <= self.q[c]: labels.append(0)       # score for y=0
                if 1-p1 <= self.q[c]: labels.append(1)     # score for y=1
                sample.append(labels)
            result.append(sample)
        return result
