import torch
import torch.nn as nn
import math

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 500000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[-1])
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, n_kv_heads=None, max_seq_len=8192):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        
        assert hidden_size % n_heads == 0
        self.head_dim = hidden_size // n_heads
        
        self.wq = nn.Linear(hidden_size, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, hidden_size, bias=False)
        
        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, x, start_pos=0, cache=None):
        B, T, C = x.shape
        
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        xq = xq.view(B, T, self.n_heads, self.head_dim)
        xk = xk.view(B, T, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, T, self.n_kv_heads, self.head_dim)
        
        req_freqs_cis = self.freqs_cis[start_pos:start_pos+T].to(x.device)
        xq, xk = apply_rotary_emb(xq, xk, req_freqs_cis)
        
        if cache is not None:
            xk, xv = cache.update(start_pos, xk, xv)
            
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            seq_len_kv = xk.shape[1]
            xk = xk[:, :, :, None, :].expand(B, seq_len_kv, self.n_kv_heads, n_rep, self.head_dim).reshape(B, seq_len_kv, self.n_heads, self.head_dim)
            xv = xv[:, :, :, None, :].expand(B, seq_len_kv, self.n_kv_heads, n_rep, self.head_dim).reshape(B, seq_len_kv, self.n_heads, self.head_dim)
            
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Causal mask applies when T > 1 and T equals KV sequence length
        is_causal = (T > 1 and T == xk.shape[2])
        out = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.wo(out)

