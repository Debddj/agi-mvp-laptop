import torch
import sentencepiece as spm

from model.cache import KVCache
from model.vision import process_image


def sample(logits, temperature=0.8, top_k=20):
    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        logits[logits < values[-1]] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def llm_plan(model, tokenizer_model, task, image_input=None, max_steps=5):
    """
    Multi-modal LLM planner that generates step-by-step action plans
    for both text-only tasks and visual/multimodal tasks.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_model)

    device = next(model.parameters()).device

    prompt = (
        "You are an intelligent multi-modal planning agent.\n"
        "Break the following task into clear, ordered steps.\n"
        "Do not solve the task.\n"
        "Only output the steps.\n\n"
        f"Task: {task}\n"
        "Steps:\n"
    )

    tokens = sp.encode(prompt)
    prompt_len = len(tokens)
    model.eval()

    image_tensor = None
    if image_input is not None:
        image_tensor = process_image(image_input).to(device)

    head_dim = model.hidden_size // model.blocks[0].attn.n_heads
    num_patches = model.vision_encoder.patch_embed.num_patches if (image_tensor is not None and hasattr(model, "vision_encoder")) else 0
    total_context = prompt_len + num_patches + 100

    caches = [
        KVCache(1, total_context, block.attn.n_kv_heads, head_dim)
        for block in model.blocks
    ]

    generated_tokens = list(tokens)

    for i in range(80):
        if i == 0:
            x = torch.tensor(tokens, device=device).unsqueeze(0)
            start_pos = 0
            curr_images = image_tensor
        else:
            x = torch.tensor([generated_tokens[-1]], device=device).unsqueeze(0)
            start_pos = (prompt_len + num_patches + i - 1) if image_tensor is not None else (prompt_len + i - 1)
            curr_images = None

        with torch.no_grad():
            logits = model(x, images=curr_images, start_pos=start_pos, caches=caches)

        next_token = sample(logits[0, -1])
        generated_tokens.append(next_token)

        if next_token == sp.eos_id():
            break

    text = sp.decode(generated_tokens)

    steps = []
    for line in text.split("\n"):
        line = line.strip()
        if line and (any(c.isdigit() for c in line) or line.startswith("-") or line.startswith("*")):
            steps.append(line)

    if not steps:
        if image_input is not None:
            steps = [
                "1. Analyze visual input features and metadata",
                "2. Execute multi-modal task pipeline",
                "3. Validate output accuracy",
            ]
        else:
            steps = [
                "1. Analyze the task requirements",
                "2. Execute the required tools",
                "3. Verify and return final result",
            ]

    return steps[:max_steps]
