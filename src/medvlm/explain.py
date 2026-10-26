import torch
from captum.attr import IntegratedGradients

def integrated_gradients_image(model,batch,class_index,n_steps=32):
    """Return a normalized absolute 2D IG attribution map for one diagnostic logit."""
    model.eval()
    image=batch["image"]
    ids=batch["input_ids"]
    mask=batch["attention_mask"]
    def forward(img):
        return model(img,ids,mask)["logits"][:,class_index]
    ig=IntegratedGradients(forward)
    attr=ig.attribute(image,baselines=torch.zeros_like(image),n_steps=n_steps)
    heat=attr.abs().sum(1)[0].detach().cpu().numpy()
    heat=heat-heat.min()
    return heat/(heat.max()+1e-8)
