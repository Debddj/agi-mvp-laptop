import argparse
import os
import torch
from agents.agent_loop import run_agent
from model.transformer import MultimodalLLM
from utils.config import load_config


def main():
    default_ckpt = os.path.abspath("checkpoints/tiny_llm.pt")

    parser = argparse.ArgumentParser(description="Multi-Modal AGI LLM Agent Launcher")
    parser.add_argument("--task", type=str, default="Analyze visual inputs and evaluate expression: 12 * (7 + 3)", help="Task prompt for the agent")
    parser.add_argument("--image", type=str, default=None, help="Path to input image (optional)")
    parser.add_argument("--checkpoint", type=str, default=default_ckpt, help="Path to model checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading Multi-Modal AGI LLM on device: {device.upper()}")

    config = load_config()

    model = MultimodalLLM(config)
    if os.path.exists(args.checkpoint):
        try:
            ckpt = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            print(f"Successfully loaded model checkpoint from '{args.checkpoint}'")
        except Exception as e:
            print(f"Warning: Could not load checkpoint ({e}). Running with initialized weights.")
    else:
        print(f"Note: Checkpoint '{args.checkpoint}' not found. Initializing model weights.")

    model.to(device)
    model.eval()

    run_agent(model, task=args.task, image_path=args.image)


if __name__ == "__main__":
    main()
