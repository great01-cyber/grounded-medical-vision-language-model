import math, torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import AutoModel

# Computes the mean representation of valid tokens while ignoring padding tokens.
# The attention mask is expanded to match the embedding dimensions, padding
# embeddings are zeroed out, and the remaining token embeddings are averaged.
def masked_mean(x, mask):
    w = mask.unsqueeze(-1).to(x.dtype)
    return (x*w).sum(1) / w.sum(1).clamp_min(1)

# Vision encoder that loads a pretrained image backbone, removes its original
# classification head, and projects the extracted image features into the
# shared multimodal embedding space. The backbone can optionally be frozen
# so that only the projection layer is trained.
class VisionEncoder(nn.Module):
    def __init__(self, name, dim, freeze=True):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=True, num_classes=0)
        self.proj = nn.Linear(self.backbone.num_features, dim)
        if freeze:
            for p in self.backbone.parameters(): p.requires_grad=False

# Extract visual features from the pretrained backbone and convert them into
# a consistent [batch, tokens, features] format. CNN feature maps are flattened
# into spatial tokens, while global 2D features are given a single-token
# dimension. The features are then projected into the shared multimodal space.

    def forward(self, x):
        z = self.backbone.forward_features(x)
        if z.ndim == 4:
            z = z.flatten(2).transpose(1,2)
        elif z.ndim == 2:
            z = z[:,None,:]

        return self.proj(z)
# Text encoder that loads a pretrained language model and projects its token
# representations into the shared multimodal embedding space. The pretrained
# backbone can optionally be frozen so that only the projection layer and
# other downstream components are updated during training.
class TextEncoder(nn.Module):
    def __init__(self, name, dim, freeze=True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name, trust_remote_code=True)
        self.proj = nn.Linear(self.backbone.config.hidden_size, dim)
        if freeze:
            for p in self.backbone.parameters(): p.requires_grad=False

    def forward(self, ids, mask):
        z = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.proj(z)

class LoRALinear(nn.Module):
    """W*x + scale*B*A*x, with W frozen and A/B trainable."""
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base=base
        for p in base.parameters(): p.requires_grad=False
        self.A=nn.Parameter(torch.empty(rank, base.in_features))
        self.B=nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale=alpha/rank
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
    def forward(self,x):
        return self.base(x) + ((x @ self.A.t()) @ self.B.t())*self.scale

class ContrastiveClassifier(nn.Module):
    """Global CLIP-style alignment plus a supervised multilabel head."""
    def __init__(self,cfg,n_labels):
        super().__init__()
        self.vision=VisionEncoder(cfg.vision_backbone,cfg.embed_dim,cfg.freeze_vision)
        self.text=TextEncoder(cfg.text_backbone,cfg.embed_dim,cfg.freeze_text)
        self.logit_scale=nn.Parameter(torch.tensor(math.log(1/.07)))
        self.head=nn.Sequential(nn.LayerNorm(2*cfg.embed_dim),nn.Dropout(cfg.dropout),
                                nn.Linear(2*cfg.embed_dim,n_labels))
    def forward(self,image,input_ids,attention_mask):
        vt=self.vision(image); tt=self.text(input_ids,attention_mask)
        vi=vt.mean(1); te=masked_mean(tt,attention_mask)
        return {"logits":self.head(torch.cat([vi,te],-1)),
                "image_embedding":F.normalize(vi,dim=-1),
                "text_embedding":F.normalize(te,dim=-1),
                "image_tokens":vt,"text_tokens":tt,
                "logit_scale":self.logit_scale.exp().clamp(max=100)}

class CrossAttentionClassifier(nn.Module):
    """Clinical-text tokens query image tokens before multilabel classification."""

    def __init__(self, cfg, n_labels, use_lora=False):
        super().__init__()

        self.vision = VisionEncoder(
            cfg.vision_backbone,
            cfg.embed_dim,
            cfg.freeze_vision
        )

        self.text = TextEncoder(
            cfg.text_backbone,
            cfg.embed_dim,
            cfg.freeze_text
        )

        self.embed_dim = cfg.embed_dim
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.embed_dim // cfg.num_heads

        if cfg.embed_dim % cfg.num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads"
            )

        # Actual Q, K and V projection matrices
        q = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        k = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        v = nn.Linear(cfg.embed_dim, cfg.embed_dim)

        # For the LoRA model, freeze the base Q/K/V matrices
        # and learn only the low-rank LoRA updates.
        if use_lora:
            q = LoRALinear(
                q,
                cfg.lora_rank,
                cfg.lora_alpha
            )

            k = LoRALinear(
                k,
                cfg.lora_rank,
                cfg.lora_alpha
            )

            v = LoRALinear(
                v,
                cfg.lora_rank,
                cfg.lora_alpha
            )

        self.q = q
        self.k = k
        self.v = v

        self.dropout = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, n_labels)

    def forward(self, image, input_ids, attention_mask):

        # --------------------------------------------------
        # 1. Encode image and text
        # --------------------------------------------------

        vt = self.vision(image)
        tt = self.text(input_ids, attention_mask)

        # vt: [B, image_tokens, embed_dim]
        # tt: [B, text_tokens, embed_dim]

        # --------------------------------------------------
        # 2. Produce Q, K and V
        # --------------------------------------------------

        Q = self.q(tt)
        K = self.k(vt)
        V = self.v(vt)

        B, T, D = Q.shape
        I = K.shape[1]

        # --------------------------------------------------
        # 3. Split embedding dimension into attention heads
        # --------------------------------------------------

        Q = Q.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            B,
            I,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            B,
            I,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # Shapes:
        #
        # Q = [B, heads, text_tokens, head_dim]
        # K = [B, heads, image_tokens, head_dim]
        # V = [B, heads, image_tokens, head_dim]

        # --------------------------------------------------
        # 4. Scaled dot-product cross-attention
        # --------------------------------------------------

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(self.head_dim)

        # [B, heads, text_tokens, image_tokens]

        aw = torch.softmax(scores, dim=-1)

        aw_drop = self.dropout(aw)

        fused = torch.matmul(
            aw_drop,
            V
        )

        # --------------------------------------------------
        # 5. Merge attention heads
        # --------------------------------------------------

        fused = fused.transpose(1, 2).contiguous()

        fused = fused.view(
            B,
            T,
            D
        )

        # --------------------------------------------------
        # 6. Residual connection
        # --------------------------------------------------

        fused = self.norm(
            fused + tt
        )

        # Ignore padding tokens when pooling text
        pooled = masked_mean(
            fused,
            attention_mask
        )

        # --------------------------------------------------
        # 7. Multilabel classification
        # --------------------------------------------------

        logits = self.head(pooled)

        return {
            "logits": logits,
            "image_tokens": vt,
            "text_tokens": tt,
            "cross_attention": aw
        }
    
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_percent": 100 * trainable / total
    }    
def trainable_parameter_breakdown(model):
    return {
        name: p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad
    }

def build_model(cfg,n_labels):
    if cfg.kind=="contrastive": return ContrastiveClassifier(cfg,n_labels)
    if cfg.kind=="cross_attention": return CrossAttentionClassifier(cfg,n_labels,False)
    if cfg.kind=="lora": return CrossAttentionClassifier(cfg,n_labels,True)
    raise ValueError(cfg.kind)
