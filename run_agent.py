from agents.agent_loop import run_agent
from model.transformer import TinyLLM


if __name__ == "__main__":
    config = {
        "vocab_size": 2000,
        "n_layers": 6,
        "n_heads": 6,
        "hidden_size": 384,
        "mlp_ratio": 4,
    }

    model = TinyLLM(config)

    run_agent(model, "Calculate 12 * (7 + 3)")
