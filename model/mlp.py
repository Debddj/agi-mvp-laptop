import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, hidden_size, mlp_ratio):
        super().__init__()
        # In Llama-style SwiGLU, hidden_dim is usually 2/3 of what it would be
        hidden_dim = int(2 * hidden_size * mlp_ratio / 3)
        # Round up to multiple of 256 for tensor core efficiency (optional but good practice)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)

        self.w1 = nn.Linear(hidden_size, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
