import pytest
import torch

from stable_diffusion.spatial_transformer import AttentionBlock, CrossAttention


def test_xattn_dims():
    torch.manual_seed(459)
    d_q = 256
    d_cross = 1024
    d_model = 512
    N = 5
    T_q = 10
    T_c = 5

    q = torch.randn(N, T_q, d_q)
    cross = torch.randn(N, T_c, d_cross)

    xattn = CrossAttention(d_q, d_model, d_cross)

    # check the cross mechanism, these should be the same
    sattn1 = CrossAttention(d_q, d_model)
    sattn2 = CrossAttention(d_q, d_model, d_q)
    sattn2.load_state_dict(sattn1.state_dict())

    out_cross = xattn(q, cross)
    out_sattn1 = sattn1(q)
    out_sattn2 = sattn2(q, q)

    assert out_cross.shape == torch.Size([N, T_q, d_q])
    assert out_sattn1.shape == torch.Size([N, T_q, d_q])
    assert (out_sattn1 == out_sattn2).all()

def test_attn_block():
    torch.manual_seed(1129)
    d_q = 256
    d_cross = 1024
    d_model = 512
    N = 5
    T_q = 10
    T_c = 5

    q = torch.randn(N, T_q, d_q)
    cross = torch.randn(N, T_c, d_cross)

    attnblock = AttentionBlock(d_q, d_cross, d_model)
    
    # check default setting
    block1 = AttentionBlock(d_q, d_q, d_model)
    block2 = AttentionBlock(d_q, d_model = d_model)
    block2.load_state_dict(block1.state_dict())

    out_cross = attnblock(q, cross)
    
    mid_context = block1.norm2(block1.attn1(block1.norm1(q)) + q)
    out_self1 = block1(q, mid_context)
    out_self2 = block2(q)
    
    assert out_cross.shape == torch.Size([N, T_q, d_q])
    assert out_self1.shape == torch.Size([N, T_q, d_q])
    assert (out_self1 == out_self2).all()
