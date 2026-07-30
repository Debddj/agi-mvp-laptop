import math
import os

import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model.cache import KVCache
from model.transformer import MultimodalLLM
from training.dataset import MultimodalDataset
from utils.config import load_config


def sample(logits, temperature=0.8, top_k=20):
    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        logits[logits < values[-1]] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def generate(model, tokenizer_model, start_text, image_tensor=None, max_new_tokens=40, temperature=0.8, top_k=20):
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_model)
    model.eval()

    device = next(model.parameters()).device
    tokens = sp.encode(start_text)
    prompt_len = len(tokens)

    head_dim = model.hidden_size // model.blocks[0].attn.n_heads
    num_patches = model.vision_encoder.patch_embed.num_patches if (image_tensor is not None and hasattr(model, "vision_encoder")) else 0
    total_context = prompt_len + num_patches + max_new_tokens

    caches = [
        KVCache(1, total_context, block.attn.n_kv_heads, head_dim)
        for block in model.blocks
    ]

    for i in range(max_new_tokens):
        if i == 0:
            x = torch.tensor(tokens, device=device).unsqueeze(0)
            start_pos = 0
            curr_images = image_tensor.to(device) if image_tensor is not None else None
        else:
            x = torch.tensor([tokens[-1]], device=device).unsqueeze(0)
            start_pos = (prompt_len + num_patches + i - 1) if image_tensor is not None else (prompt_len + i - 1)
            curr_images = None

        with torch.no_grad():
            logits = model(x, images=curr_images, start_pos=start_pos, caches=caches)

        next_token = sample(logits[0, -1], temperature, top_k)
        tokens.append(next_token)

        if next_token == sp.eos_id():
            break

    return sp.decode(tokens)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr=1e-5):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, optimizer, scaler, scheduler, config, device):
    model.train()
    total_loss = 0.0
    use_amp = config.get("use_amp", torch.cuda.is_available())
    loss_fn = nn.CrossEntropyLoss()

    for step, batch in enumerate(dataloader):
        if len(batch) == 3:
            x, y, images = batch
            images = images.to(device)
        else:
            x, y = batch
            images = None

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp and device == "cuda", dtype=torch.float16):
            logits = model(x, images=images)
            vocab_size = logits.shape[-1]
            # Slice logits to match the text target sequence length y
            text_logits = logits[:, -y.shape[1]:, :]
            loss = loss_fn(text_logits.reshape(-1, vocab_size), y.view(-1))

        if use_amp and device == "cuda":
            scaler.scale(loss).backward()
            if config.get("grad_clip", 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

        if step % 20 == 0:
            print(f"Step {step:03d} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    return total_loss / max(1, len(dataloader))


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("==========================================")
    print("MULTI-MODAL AGI LLM TRAINING ENGINE")
    print(f"Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("==========================================\n")

    config = load_config()
    model = MultimodalLLM(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.01)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and config.get("use_amp", True)))

    dataset = MultimodalDataset(
        text_file="data/sample.txt",
        tokenizer_model="tokenizer.model",
        seq_len=config["seq_len"],
        image_size=config.get("image_size", 224)
    )

    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = config.get("warmup_steps", int(0.1 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    for epoch in range(config["epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{config['epochs']} ---")
        train_loss = train_epoch(model, train_loader, optimizer, scaler, scheduler, config, device)
        print(f"Epoch {epoch + 1} Complete | Train Loss: {train_loss:.4f}")

    ckpt_dir = os.path.abspath("checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_path = os.path.join(ckpt_dir, "tiny_llm.pt")
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
    }, checkpoint_path)
    print(f"\nSaved Multi-Modal Model checkpoint to {checkpoint_path}")

    print("\n--- SAMPLE GENERATION TEST ---")
    generated = generate(model, "tokenizer.model", "Artificial intelligence", max_new_tokens=30)
    print(f"Output: {generated}\n")


if __name__ == "__main__":
    main()
