import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Sequential):
    """Feed forward module for the attention block.  Has two  linear layers with a GeLU and dropout layer in between.

    Args:
        d_in (int): Input dimension to the first linear layer.
        d_out (int): Output dimension of the second linear layer.
        mult (int): Multiplier (of the input dimension) for the hidden dimension. Default: 4
        dropout (float): Dropout rate. Default: 0.1
    """

    def __init__(self, d_in, d_out, mult=4, dropout=0.1):
        super().__init__()

        self.proj_in = nn.Linear(d_in, int(d_in * mult))
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(int(d_in * mult), d_out)


# Cross attention module, from scratch
class CrossAttention(nn.Module):
    """Basic cross attention module.

    Args:
        d_q (int): Input dimension of the query.
        d_model (int): Inner dimension of the QKV projection layers. Default: 512
        d_cross (int): Input dimension of the key and value inputs, for cross attention. Default: None
        n_heads (int): Number of attention heads. Default: 8
        dropout (float): Dropout rate. Default: 0.0
    """

    def __init__(self, d_q, d_model=512, d_cross=None, n_heads=8, dropout=0.0):
        super().__init__()

        assert d_model % n_heads == 0, f"n_heads {n_heads} must divide d_model {d_model}"

        if d_cross is None:
            d_cross = d_q

        self.proj_q = nn.Linear(d_q, d_model, bias=False)
        self.proj_k = nn.Linear(d_cross, d_model, bias=False)
        self.proj_v = nn.Linear(d_cross, d_model, bias=False)

        self.proj_out = nn.Linear(d_model, d_q)
        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads

    def forward(self, x, context=None, mask=None):
        # prevent einops broadcasting
        if context is not None:
            assert (
                x.shape[0] == context.shape[0]
            ), f"Batch size of x and context must match, found {x.shape[0]} and {context.shape[0]}"

        if context is None:
            context = x

        q = self.proj_q(x)
        k = self.proj_k(context)
        v = self.proj_v(context)

        # at this point we've already flattened the h/w of the input
        q = einops.rearrange(q, "b n (h d) -> b h n d", h=self.n_heads)
        k = einops.rearrange(k, "b m (h d) -> b h m d", h=self.n_heads)
        v = einops.rearrange(v, "b m (h d) -> b h m d", h=self.n_heads)

        qk = einops.einsum(q, k, "b h n d, b h m d -> b h n m") / (q.shape[-1] ** 0.5)

        if mask is not None:
            # mask initially of shape b x m, need to expand to b x h x 1 x m
            mask = einops.repeat(mask, "b m -> b h () m", h=self.n_heads)
            min_value = -torch.finfo(qk.dtype).max
            qk.masked_fill_(~mask, min_value)

        qk = F.softmax(qk, dim=-1)
        out = einops.einsum(qk, v, "b h n m, b h m d -> b h n d")
        out = einops.rearrange(out, "b h n d -> b n (h d)")

        out = self.dropout(self.proj_out(out))

        return out


class AttentionBlock(nn.Module):
    """Wrapper for two cross-attention blocks followed by a feedforward layer.

    Args:
        d_q (int): Input dimension of the query.
        d_cross (int): Input dimension of the key and value inputs, for cross attention. Default: None
        d_model (int): Inner dimension of the QKV projection layers. Default: 512
        n_heads (int): Number of attention heads. Default: 8
        dropout (float): Dropout rate. Default: 0.0
    """

    def __init__(self, d_q, d_cross=None, d_model=512, n_heads=8, dropout=0.0):
        super().__init__()

        if d_cross is None:
            d_cross = d_q

        self.attn1 = CrossAttention(d_q, d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.attn2 = CrossAttention(
            d_q, d_cross=d_cross, d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.ff = FeedForward(d_q, d_q, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_q)
        self.norm2 = nn.LayerNorm(d_q)
        self.norm3 = nn.LayerNorm(d_q)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """Spatial transformer module for the UNet architecture.  Contains cross-attention layers that attend over the spatial dimensions of an image, while ingesting cross-attention embeddings from e.g. a text embedding model.

    Args:
        in_channels (int): Number of input channels.
        d_q (int): Input dimension of the query.
        d_cross (int): Input dimension of the key and value inputs, for cross attention. Default: None
        d_model (int): Inner dimension of the QKV projection layers. Default: 512
        n_heads (int): Number of attention heads. Default: 8
        dropout (float): Dropout rate. Default: 0.0
        depth (int): Number of attention blocks. Default: 1
    """

    def __init__(
        self, in_channels, d_q, d_cross=None, d_model=512, n_heads=8, dropout=0.0, depth=1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.d_q = d_q
        self.d_cross = d_cross
        self.d_model = d_model
        self.n_heads = n_heads

        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )

        self.conv_in = nn.Conv2d(self.in_channels, d_q, kernel_size=1, stride=1, padding=0)

        self.blocks = nn.ModuleList(
            [AttentionBlock(d_q, d_cross, d_model, n_heads, dropout) for _ in range(depth)]
        )

        self.conv_out = nn.Conv2d(d_q, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context=None):
        x_in = x

        x = self.norm(x)
        x = self.conv_in(x)  # B, d_q, H, W

        b, d_q, h, w = x.shape

        x = einops.rearrange(x, "b c h w -> b (h w) c")  # attention mechanism expects B T C

        for b in self.blocks:
            x = b(x, context)

        x = einops.rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.conv_out(x)
        return x + x_in
