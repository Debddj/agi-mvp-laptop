import torch
import torch.nn as nn
from model.transformer import TinyLLM

config = {
    "vocab_size": 8000,
    "n_layers": 6,
    "n_heads": 6,
    "hidden_size": 384,
    "mlp_ratio": 4
}

model = TinyLLM(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

# Fake data for sanity check
x = torch.randint(0, 8000, (2, 128))
y = x.clone()

for step in range(50):
    logits = model(x)
    loss = loss_fn(logits.view(-1, 8000), y.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Step {step} | Loss: {loss.item():.4f}")
