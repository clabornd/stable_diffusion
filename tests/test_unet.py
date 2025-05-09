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
    T_c = 5

    myunet = UNET(
        channels_in = channels_in, 
        channels_model = d_model,
        channels_out = channels_out,
        t_emb_dim = t_emb_dim,
        context_dim = d_cross,
        d_model = d_model
    )

    cross = torch.randn(N, T_c, d_cross)
    dummy_input = torch.randn(N, channels_in, imsize, imsize)
    t_steps = torch.randint(0, 1000, size = (N,))
    out = myunet(dummy_input, timesteps = t_steps, context = cross)

    assert out.shape == (N, channels_in, imsize, imsize)

    with pytest.raises(AssertionError):
        myunet(dummy_input, timesteps = t_steps, context = None)

    myunet_nocross = UNET(
        channels_in = channels_in, 
        channels_model = d_model,
        channels_out = channels_out,
        t_emb_dim = t_emb_dim,
        context_dim = None,
        d_model = d_model
    )

    assert myunet_nocross.down_blocks[1][0].time_emb[-1].in_features == t_emb_dim

    with pytest.raises(AssertionError):
        myunet_nocross(dummy_input, timesteps = t_steps, context = cross)

    myunet_notime = UNET(
        channels_in = channels_in, 
        channels_model = d_model,
        channels_out = channels_out,
        t_emb_dim = None,
        context_dim = d_cross,
        d_model = d_model
    )

    assert myunet_notime.down_blocks[1][0].time_emb[-1].in_features == d_model

    out = myunet_notime(dummy_input, timesteps = t_steps, context = cross)
