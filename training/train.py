import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sentencepiece as spm

from model.transformer import TinyLLM
from training.dataset import TextDataset


# -------------------------------------------------
# Text generation (Day 7)
# -------------------------------------------------
def generate(model, tokenizer_model, start_text, max_new_tokens=40):
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_model)

    model.eval()

    tokens = sp.encode(start_text)

    for _ in range(max_new_tokens):
        x = torch.tensor(tokens).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
        next_token = logits[0, -1].argmax().item()
        tokens.append(next_token)

    return sp.decode(tokens)


# -------------------------------------------------
# Training
# -------------------------------------------------
def main():
    config = {
        "vocab_size": 2000,
        "n_layers": 6,
        "n_heads": 6,
        "hidden_size": 384,
        "mlp_ratio": 4,
        "seq_len": 128,
        "batch_size": 2,
        "lr": 3e-4,
        "epochs": 2,
    }

    device = "cpu"

    # Model
    model = TinyLLM(config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()

    # Dataset
    dataset = TextDataset(
        text_file="data/sample.txt",
        tokenizer_model="tokenizer.model",
        seq_len=config["seq_len"],
    )

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    # ------------------------
    # Training loop
    # ------------------------
    model.train()
    for epoch in range(config["epochs"]):
        total_loss = 0.0

        for step, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(
                logits.view(-1, config["vocab_size"]),
                y.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 50 == 0:
                print(
                    f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}"
                )

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} finished | Avg Loss: {avg_loss:.4f}")

    print("\nTraining complete.")

    # ------------------------
    # Day 7: Sample generation
    # ------------------------
    print("\n--- SAMPLE GENERATION ---")
    output = generate(
        model,
        tokenizer_model="tokenizer.model",
        start_text="Artificial intelligence",
        max_new_tokens=40,
    )
    print(output)


if __name__ == "__main__":
    main()
