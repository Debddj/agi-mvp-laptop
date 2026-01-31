import torch.nn as nn
from .attention import CausalSelfAttention
from .mlp import MLP
from .norms import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, mlp_ratio):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, n_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = MLP(hidden_size, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
class TinyLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config["hidden_size"],
                config["n_heads"],
                config["mlp_ratio"]
            )
            for _ in range(config["n_layers"])
        ])
        self.norm = RMSNorm(config["hidden_size"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)
