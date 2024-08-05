import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from .resblocks import conv_nd


class FeedForward(nn.Sequential):
    def __init__(self, d_in, d_out, mult=4, dropout=0.1):
        super().__init__()

        self.proj_in = nn.Linear(d_in, int(d_in*mult))
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(int(d_in*mult), d_out)

# Cross attention module, from scratch
class CrossAttention(nn.Module):
    def __init__(self, d_q, d_model=512, d_cross = None, n_heads=8, dropout=0.0):
        super().__init__()
        
        assert d_model % n_heads == 0, f"n_heads {n_heads} must divide d_model {d_model}"
        
        if d_cross is None:
            d_cross = d_q

        self.proj_q = nn.Linear(d_q, d_model, bias = False)
        self.proj_k = nn.Linear(d_cross, d_model, bias = False)
        self.proj_v = nn.Linear(d_cross, d_model, bias = False)

        self.proj_out = nn.Linear(d_model, d_q)
        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads

    def forward(self, x, context = None, mask = None):
        if context is None:
            context = x
        
        q = self.proj_q(x)
        k = self.proj_k(context)
        v = self.proj_v(context)

        # at this point we've already flattened the h/w of the input
        q = einops.rearrange(q, 'b n (h d) -> b h n d', h = self.n_heads)
        k = einops.rearrange(k, 'b m (h d) -> b h m d', h = self.n_heads)
        v = einops.rearrange(v, 'b m (h d) -> b h m d', h = self.n_heads)

        qk = einops.einsum(q, k, 'b h n d, b h m d -> b h n m') / (q.shape[-1] ** 0.5)

        if mask is not None:
            # mask initially of shape b x m, need to expand to b x h x 1 x m
            mask = einops.repeat(mask, 'b m -> b h () m', h = self.n_heads)
            min_value = -torch.finfo(qk.dtype).max
            qk.masked_fill_(~mask, min_value)

        qk = F.softmax(qk, dim = -1)
        out = einops.einsum(qk, v, 'b h n m, b h m d -> b h n d')
        out = einops.rearrange(out, 'b h n d -> b n (h d)')

        out = self.dropout(self.proj_out(out))

        return out

        
class AttentionBlock(nn.Module):
    def __init__(self, d_q, d_cross = None, d_model = 512, n_heads = 8, dropout = 0.0):
        super().__init__()

        if d_cross is None:
            d_cross = d_q
        
        self.attn1 = CrossAttention(d_q, d_model = d_model, n_heads=n_heads, dropout=dropout)
        self.attn2 = CrossAttention(d_q, d_cross = d_cross, d_model = d_model, n_heads=n_heads, dropout=dropout)
        self.ff = FeedForward(d_q, d_q, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_q)
        self.norm2 = nn.LayerNorm(d_q)
        self.norm3 = nn.LayerNorm(d_q)

    def forward(self, x, context = None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context = context) + x
        x = self.ff(self.norm3(x)) + x
        return x
        

# attnblock = AttentionBlock(512, d_cross = 1024, d_model = 512)

# q = torch.randn(10, 16, 512)
# context = torch.randn(10, 5, 1024)

# attnblock(q, context).shape