import pytest
import torch

from stable_diffusion.unet import UNET


def test_unet():
    channels_out = 3
    channels_in = 3
    imsize = 64
    t_emb_dim = 16
    d_cross = 16
    d_model = 32
    N = 1
    T_q = 10
    T_c = 5

    myunet = UNET(
        channels_in = channels_in, 
        channels_model = d_model,
        channels_out = channels_out,
        t_emb_dim = t_emb_dim,
        context_dim = d_cross,
        d_model = d_model
    )

    q = torch.randn(N, T_q, channels_out)
    cross = torch.randn(N, T_c, d_cross)
    dummy_input = torch.randn(N, channels_in, imsize, imsize)
    t_emb = torch.randn(N, t_emb_dim)

    out = myunet(dummy_input, timesteps = t_emb, context = cross)

    assert out.shape == (N, channels_in, imsize, imsize)