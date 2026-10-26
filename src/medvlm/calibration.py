import torch
import torch.nn.functional as F

class PerClassTemperatureScaler:
    """Positive per-label temperatures fitted only on a held-out calibration split."""
    def __init__(self,n_classes):
        self.log_t=torch.zeros(n_classes,requires_grad=True)

    def fit(self,logits,y,mask,max_iter=100):
        logits,y,mask=[x.detach().float().cpu() for x in (logits,y,mask)]
        self.log_t=torch.zeros(logits.shape[1],requires_grad=True)
        opt=torch.optim.LBFGS([self.log_t],lr=.1,max_iter=max_iter,line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            t=self.log_t.exp().clamp(.05,20)
            loss=F.binary_cross_entropy_with_logits(logits/t,y,reduction="none")
            loss=(loss*mask).sum()/mask.sum().clamp_min(1)
            loss.backward()
            return loss
        opt.step(closure)
        return self

    @property
    def temperature(self):
        return self.log_t.detach().exp().clamp(.05,20)

    def transform(self,logits):
        return logits/self.temperature.to(logits.device)
