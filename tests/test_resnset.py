import pytest
import torch

from stable_diffusion.resblocks import ResBlock


def test_resblock_dims():
    res_down = ResBlock(32, d_emb = 64, channels_out = 64, resample = 'down')
    res_up = ResBlock(32, d_emb = 64, channels_out = 64, resample = 'up')
    res_none = ResBlock(32, d_emb = 32, channels_out=32)

    dummy_input = torch.randn(1, 32, 64, 64)
    emb64 = torch.randn(1, 64)
    emb32 = torch.randn(1, 32)

    output_down = res_down(dummy_input, emb64)
    output_up = res_up(dummy_input, emb64)
    output_none = res_none(dummy_input, emb32)

    assert output_down.shape == (1, 64, 32, 32)
    assert output_up.shape == (1, 64, 128, 128)
    assert output_none.shape == (1, 32, 64, 64)