import torch
import torch.nn as nn
from model.attention import CausalSelfAttention
from model.mlp import MLP
from model.norms import RMSNorm
from model.vision import VisionEncoder


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, n_kv_heads, mlp_ratio, max_seq_len=8192):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, n_heads, n_kv_heads, max_seq_len=max_seq_len)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = MLP(hidden_size, mlp_ratio)

    def forward(self, x, start_pos=0, cache=None):
        x = x + self.attn(self.norm1(x), start_pos, cache)
        x = x + self.mlp(self.norm2(x))
        return x


class MultimodalLLM(nn.Module):
    """
    Full-fledged Multi-Modal Transformer LM architecture supporting both text tokens
    and visual image patch embeddings in a unified causal sequence.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.max_seq_len = config.get("max_seq_len", 8192)

        # Text embeddings
        self.embed = nn.Embedding(self.vocab_size, self.hidden_size)
        self.pos_embed = nn.Embedding(self.max_seq_len, self.hidden_size)

        # Vision Encoder module
        image_size = config.get("image_size", 224)
        patch_size = config.get("patch_size", 16)
        vision_hidden_dim = config.get("vision_hidden_dim", self.hidden_size)
        
        self.vision_encoder = VisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            vision_hidden_dim=vision_hidden_dim,
            llm_hidden_dim=self.hidden_size,
            n_layers=config.get("vision_layers", 4),
            n_heads=config.get("vision_heads", config.get("n_heads", 6))
        )

        # Transformer layers
        self.blocks = nn.ModuleList([
            TransformerBlock(
                self.hidden_size,
                config["n_heads"],
                config.get("n_kv_heads", config["n_heads"]),
                config["mlp_ratio"],
                self.max_seq_len
            )
            for _ in range(config["n_layers"])
        ])
        
        self.norm = RMSNorm(self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed.weight

    def forward(self, x, images=None, start_pos=0, caches=None):
        """
        x: [B, T] token IDs
        images: Optional [B, 3, H, W] image tensor
        start_pos: current position offset in sequence (for KV cache decoding)
        caches: list of KVCache objects for each block
        """
        B, T = x.shape
        text_embeds = self.embed(x)  # [B, T, hidden_size]

        if images is not None and start_pos == 0:
            # Multi-modal prefill step: prepend vision patch tokens ahead of text prompt
            vision_embeds = self.vision_encoder(images)  # [B, N_patches, hidden_size]
            x_embeds = torch.cat([vision_embeds, text_embeds], dim=1)  # [B, N_patches + T, hidden_size]
        else:
            x_embeds = text_embeds

        B, Total_T, H = x_embeds.shape
        pos = torch.arange(start_pos, start_pos + Total_T, device=x_embeds.device).unsqueeze(0)
        pos = torch.clamp(pos, max=self.max_seq_len - 1)
        
        x_embeds = x_embeds + self.pos_embed(pos)

        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            x_embeds = block(x_embeds, start_pos, cache)

        x_embeds = self.norm(x_embeds)
        logits = self.lm_head(x_embeds)
        return logits


# Backwards compatibility alias
TinyLLM = MultimodalLLM
