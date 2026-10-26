import torch, torch.nn.functional as F

#Compute masked binary cross-entropy for multi-label diagnosis.
# Only labels marked as valid by the mask contribute to the loss.
def masked_bce(logits,y,mask):
    z=F.binary_cross_entropy_with_logits(logits,y,reduction="none")
    return (z*mask).sum()/mask.sum().clamp_min(1)

# Compute CLIP-style symmetric InfoNCE loss. Each image is compared with
# every text embedding in the batch, with the matching image-text pairs
# located on the diagonal of the similarity matrix. Cross-entropy is
# calculated in both image-to-text and text-to-image directions and averaged.
def symmetric_infonce(image_embedding,text_embedding,logit_scale):
    s=logit_scale*(image_embedding@text_embedding.T)
    target=torch.arange(s.shape[0],device=s.device)
    return .5*(F.cross_entropy(s,target)+F.cross_entropy(s.T,target))
