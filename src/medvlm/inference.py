import numpy as np
from .selective import abstain_mask,sample_confidence,sample_entropy

def research_output(probabilities,labels,conformal_sets=None,confidence_threshold=.8,entropy_threshold=.55):
    p=np.asarray(probabilities,float)[None,:]
    sets=[conformal_sets] if conformal_sets is not None else None
    abst=bool(abstain_mask(p,sets,confidence_threshold,entropy_threshold)[0])
    return {
      "status":"ABSTAIN — insufficient confidence; refer to clinician." if abst else "MODEL OUTPUT — research use only.",
      "prediction":[{"finding":n,"probability":float(v),"present":bool(v>=.5)} for n,v in zip(labels,p[0])],
      "uncertainty":{"mean_binary_confidence":float(sample_confidence(p)[0]),
                     "mean_predictive_entropy":float(sample_entropy(p)[0]),
                     "conformal_sets":conformal_sets},
      "explanation_policy":"Generate a concise evidence summary from measured predictions/attributions; do not claim hidden chain-of-thought."
    }
