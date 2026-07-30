import torch
import torch.nn as nn

class KVCache(nn.Module):
    k_cache: torch.Tensor
    v_cache: torch.Tensor

    def __init__(self, max_batch_size, max_seq_len, n_kv_heads, head_dim):
        super().__init__()
        self.register_buffer("k_cache", torch.zeros(max_batch_size, max_seq_len, n_kv_heads, head_dim))
        self.register_buffer("v_cache", torch.zeros(max_batch_size, max_seq_len, n_kv_heads, head_dim))

    def update(self, start_pos, xk, xv):
        B, T = xk.size(0), xk.size(1)
        self.k_cache[:B, start_pos:start_pos+T] = xk
        self.v_cache[:B, start_pos:start_pos+T] = xv
        
        return self.k_cache[:B, :start_pos+T], self.v_cache[:B, :start_pos+T]
        
    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
