import os
import yaml

DEFAULT_CONFIG = {
    # Model parameters
    "vocab_size": 2000,
    "n_layers": 6,
    "n_heads": 6,
    "n_kv_heads": 6,
    "hidden_size": 384,
    "mlp_ratio": 4,
    "max_seq_len": 8192,
    "image_size": 224,
    "patch_size": 16,
    "vision_hidden_dim": 384,
    "vision_layers": 4,
    "vision_heads": 6,
    # Training parameters
    "seq_len": 128,
    "batch_size": 2,
    "lr": 0.0003,
    "epochs": 2,
    "grad_clip": 1.0,
    "use_amp": True,
    "warmup_steps": 10,
    "weight_decay": 0.01,
}


def load_config(path="configs/tiny_llm.yaml"):
    if not os.path.exists(path):
        return dict(DEFAULT_CONFIG)

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except Exception:
        raw = None

    config = dict(DEFAULT_CONFIG)

    if raw and isinstance(raw, dict):
        if "model" in raw and isinstance(raw["model"], dict):
            config.update(raw["model"])
        if "training" in raw and isinstance(raw["training"], dict):
            config.update(raw["training"])

    # Ensure correct data types for float and int fields
    for float_field in ("lr", "grad_clip", "weight_decay"):
        if float_field in config:
            config[float_field] = float(config[float_field])

    for int_field in ("vocab_size", "n_layers", "n_heads", "n_kv_heads", "hidden_size", "mlp_ratio", "max_seq_len", "image_size", "patch_size", "vision_hidden_dim", "vision_layers", "vision_heads", "seq_len", "batch_size", "epochs", "warmup_steps"):
        if int_field in config:
            config[int_field] = int(config[int_field])

    return config
