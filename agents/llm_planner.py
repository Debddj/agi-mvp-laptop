import torch
import sentencepiece as spm


def llm_plan(model, tokenizer_model, task, max_steps=5):
    """
    Uses the LLM ONLY to generate a plan.
    No execution, no tools.
    """

    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_model)

    prompt = (
        "You are an intelligent planning agent.\n"
        "Break the following task into clear, ordered steps.\n"
        "Do not solve the task.\n"
        "Only output the steps.\n\n"
        f"Task: {task}\n"
        "Steps:\n"
    )

    tokens = sp.encode(prompt)
    model.eval()

    for _ in range(80):
        x = torch.tensor(tokens).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
        next_token = logits[0, -1].argmax().item()
        tokens.append(next_token)

    text = sp.decode(tokens)

    # Parse steps safely
    steps = []
    for line in text.split("\n"):
        line = line.strip()
        if line and any(c.isdigit() for c in line):
            steps.append(line)

    if not steps:
        steps = [
            "Analyze the task",
            "Execute the solution",
            "Verify the result",
        ]

    return steps[:max_steps]
