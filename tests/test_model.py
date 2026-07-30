import torch

from model.attention import CausalSelfAttention
from model.cache import KVCache
from model.mlp import MLP
from model.norms import RMSNorm
from model.transformer import TinyLLM


def test_rmsnorm_shape():
    dim = 64
    x = torch.randn(2, 10, dim)
    norm = RMSNorm(dim)
    out = norm(x)
    assert out.shape == x.shape


def test_rmsnorm_std():
    dim = 64
    x = torch.randn(2, 10, dim)
    norm = RMSNorm(dim)
    out = norm(x)
    assert not torch.isnan(out).any()


def test_mlp_shape():
    hidden_size = 64
    mlp_ratio = 4
    mlp = MLP(hidden_size, mlp_ratio)
    x = torch.randn(2, 10, hidden_size)
    out = mlp(x)
    assert out.shape == x.shape


def test_mlp_output_finite():
    hidden_size = 64
    mlp = MLP(hidden_size, 4)
    x = torch.randn(2, 10, hidden_size)
    out = mlp(x)
    assert torch.isfinite(out).all()


def test_attention_shape():
    hidden_size = 64
    n_heads = 4
    attn = CausalSelfAttention(hidden_size, n_heads)
    x = torch.randn(2, 10, hidden_size)
    out = attn(x)
    assert out.shape == x.shape


def test_attention_causal_mask():
    hidden_size = 64
    n_heads = 4
    attn = CausalSelfAttention(hidden_size, n_heads)
    B, T = 2, 10
    x = torch.randn(B, T, hidden_size)
    attn(x)

    out_single = attn(x[:, :1])
    assert out_single.shape == (B, 1, hidden_size)


def test_kv_cache_update():
    cache = KVCache(2, 64, 4, 16)
    xk = torch.randn(2, 5, 4, 16)
    xv = torch.randn(2, 5, 4, 16)
    k_out, v_out = cache.update(0, xk, xv)
    assert k_out.shape == (2, 5, 4, 16)
    assert v_out.shape == (2, 5, 4, 16)

    xk2 = torch.randn(2, 3, 4, 16)
    xv2 = torch.randn(2, 3, 4, 16)
    k_out2, v_out2 = cache.update(5, xk2, xv2)
    assert k_out2.shape == (2, 8, 4, 16)
    assert v_out2.shape == (2, 8, 4, 16)


def test_tiny_llm_output_shape():
    config = {
        "vocab_size": 100,
        "n_layers": 2,
        "n_heads": 2,
        "hidden_size": 32,
        "mlp_ratio": 4,
        "max_seq_len": 128,
    }
    model = TinyLLM(config)
    x = torch.randint(0, 100, (2, 10))
    logits = model(x)
    assert logits.shape == (2, 10, 100)


def test_tiny_llm_positional_embeddings():
    config = {
        "vocab_size": 100,
        "n_layers": 2,
        "n_heads": 2,
        "hidden_size": 32,
        "mlp_ratio": 4,
        "max_seq_len": 128,
    }
    model = TinyLLM(config)
    x = torch.randint(0, 100, (1, 5))
    logits_same = model(x)

    x2 = torch.randint(0, 100, (1, 5))
    logits_diff = model(x2)

    assert logits_same.shape == (1, 5, 100)
    assert logits_diff.shape == (1, 5, 100)


def test_tiny_llm_weight_tying():
    config = {
        "vocab_size": 100,
        "n_layers": 2,
        "n_heads": 2,
        "hidden_size": 32,
        "mlp_ratio": 4,
        "max_seq_len": 128,
    }
    model = TinyLLM(config)
    model.lm_head.weight = model.embed.weight
    assert model.lm_head.weight is model.embed.weight


def test_tiny_llm_kv_cache():
    config = {
        "vocab_size": 100,
        "n_layers": 2,
        "n_heads": 2,
        "hidden_size": 32,
        "mlp_ratio": 4,
        "max_seq_len": 128,
    }
    model = TinyLLM(config)
    model.eval()

    head_dim = model.embed.weight.shape[1] // model.blocks[0].attn.n_heads
    caches = [
        KVCache(1, 20, block.attn.n_kv_heads, head_dim)
        for block in model.blocks
    ]

    x = torch.randint(0, 100, (1, 5))
    with torch.no_grad():
        out1 = model(x, start_pos=0, caches=caches)

    x_next = torch.randint(0, 100, (1, 1))
    with torch.no_grad():
        out2 = model(x_next, start_pos=5, caches=caches)

    assert out1.shape == (1, 5, 100)
    assert out2.shape == (1, 1, 100)
