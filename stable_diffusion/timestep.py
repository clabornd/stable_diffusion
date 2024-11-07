import torch


def time_embeddings(t, out_dim, max_period=10000):
    half = out_dim // 2
    denom = torch.exp(
        -torch.tensor(max_period).log() * torch.arange(0, half, dtype=torch.float32) / half
    )
    phases = t[:, None] * denom[None]

    # concatentate to form a tensor of shape B x out_dim
    out_emb = torch.cat([phases.sin(), phases.cos()], dim=-1)

    # if out_dim is odd, add a zero column
    if out_dim % 2:
        out_emb = torch.cat([out_emb, torch.zeros_like(out_emb[:, :1])], dim=-1)

    return out_emb
