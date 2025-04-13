import torch


def time_embeddings(t: torch.tensor, out_dim: int, max_period: int = 10000):
    """Create timestep embeddings from a vector of ints representing timesteps
    Args:
        t (torch.Tensor): Tensor of shape (B,) containing timesteps
        out_dim (int): Dimension of the output embeddings
        max_period (int): Maximum period for the sine and cosine functions

    Returns:
        torch.Tensor: Tensor of shape (B, out_dim) containing the timestep embeddings
    """
    half = out_dim // 2
    denom = torch.exp(
        -torch.tensor(max_period).log() * torch.arange(0, half, dtype=torch.float32) / half
    )

    denom = denom.to(t.device)

    phases = t[:, None] * denom[None]

    # concatentate to form a tensor of shape B x out_dim
    out_emb = torch.cat([phases.sin(), phases.cos()], dim=-1)

    # if out_dim is odd, add a zero column
    if out_dim % 2:
        out_emb = torch.cat([out_emb, torch.zeros_like(out_emb[:, :1])], dim=-1)

    return out_emb
