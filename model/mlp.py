import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden_size, mlp_ratio):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * mlp_ratio)
        self.fc2 = nn.Linear(hidden_size * mlp_ratio, hidden_size)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
