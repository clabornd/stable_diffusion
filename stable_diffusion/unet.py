import torch
import torch.nn as nn

from .resblocks import ResBlock
from .spatial_transformer import SpatialTransformer
from .timestep import time_embeddings


class EmbeddingWrapper(nn.Sequential):
    """
    Wrapper for a sequence of layers that take an input tensor and optionally either a tensor of time embeddings or context embeddings.
    """

    def forward(self, x: torch.tensor, t_emb: torch.tensor = None, context: torch.tensor = None):
        """
        Args:
            x (torch.tensor): main input image tensor B x C x H x W
            t_emb (torch.tensor): time embedding B x t_emb_dim
            context (torch.tensor): context tensor B x d_cross to be passed as context to cross-attention mechanism.

        Returns:
            torch.tensor: output tensor
        """
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)

        return x


class UNET(nn.Module):
    """A simpler implementation of the UNET at https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/openaimodel.py
    Here I force the use of spatial attention when adding the guidance layers.

    Args:
        channels_in (int): number of input channels
        channels_model (int): number of initial channels which is then multiplied by the values in `channel_mults`
        channels_out (int): number of output channels
        context_dim (int): context dimension when performing guided diffusion
        d_model (int): embedding dimension of the attention layers
        t_emb_dim (int): time embedding dimension
        channel_mults (list): list of channel multipliers which will determine the number of channels at each block depending on `channels_model`
        attention_resolutions (list): list of attention resolutions where attention is applied
        dropout (float): dropout rate
    """

    def __init__(
        self,
        channels_in: int,
        channels_model: int,
        channels_out: int,
        context_dim: int,
        d_model: int,
        t_emb_dim: int = None,
        channel_mults: list[int] = [1, 2, 4, 8],
        attention_resolutions: list[int] = [2, 4],
        dropout: float = 0.0,
    ):
        super().__init__()

        self.channel_mults = channel_mults
        self.channels_in = channels_in
        self.channels_model = channels_model
        self.context_dim = context_dim
        self.t_emb_dim = t_emb_dim or channels_model

        # will fill up the downsampling and upsampling trunks in for loops
        self.down_blocks = nn.ModuleList(
            [EmbeddingWrapper(nn.Conv2d(channels_in, channels_model, kernel_size=3, padding=1))]
        )
        self.up_blocks = nn.ModuleList()

        track_chans = [channels_model]
        ch_in = channels_model

        # create each downsampling trunk
        for i, mult in enumerate(channel_mults):
            # lets assume 1 depth for each downsampling layer
            # append the reblock first, then spatial transformer
            # what is timestep dimension???? t_emb -> d_t -> d_model
            ch_out = channels_model * mult

            resblock = ResBlock(channels_in=ch_in, d_emb=self.t_emb_dim, channels_out=ch_out)
            layers = [resblock]

            if i in attention_resolutions:
                sp_transformer = SpatialTransformer(
                    in_channels=ch_out,
                    d_q=ch_out,
                    d_cross=context_dim if context_dim else ch_out,
                    d_model=d_model,
                    dropout=dropout,
                    n_heads=2,
                )
                layers.append(sp_transformer)

            self.down_blocks.append(EmbeddingWrapper(*layers))

            track_chans.append(ch_out)

            # downsample after every mult except the last
            if i != len(channel_mults) - 1:
                res_ds = ResBlock(
                    ch_out, d_emb=self.t_emb_dim, channels_out=ch_out, resample="down"
                )
                self.down_blocks.append(EmbeddingWrapper(res_ds))
                track_chans.append(ch_out)
                ch_in = ch_out

        # middle block, this is Res, Attention, Res
        # ch_out is the last channel dimension for constructing the downsampling layers
        self.middle_block = EmbeddingWrapper(
            ResBlock(channels_in=ch_out, d_emb=self.t_emb_dim),
            SpatialTransformer(
                in_channels=ch_out,
                d_q=ch_out,
                d_cross=context_dim if context_dim else ch_out,
                d_model=d_model,
                dropout=dropout,
            ),
            ResBlock(channels_in=ch_out, d_emb=self.t_emb_dim),
        )

        # upsampling block
        # this block has 2x the channels, why?  Because of the UNET architecture, we concatenate the channels from the corresponding layer of the downsample section.
        # There is also an additional res + attention block, this additional block 'matches' the channel dimension of the downsampling module in the downsampling trunk.
        for i, mult in reversed(list(enumerate(channel_mults))):
            # We assume there's two resblocks here for simplicity

            # first res block
            down_ch = track_chans.pop()

            # We have two of these, one that matches the Res + Transformer block, and another that matches the downsampling block

            # first res block
            res1 = ResBlock(
                ch_out + down_ch, d_emb=self.t_emb_dim, channels_out=channels_model * mult
            )

            # this block will output this many channels, we set it here since we want the next iteration to start the channels of the previous block.
            ch_out = channels_model * mult

            layers = [res1]

            if i in attention_resolutions:
                sp_trf = SpatialTransformer(
                    in_channels=channels_model * mult,
                    d_q=channels_model * mult,
                    d_cross=context_dim if context_dim else channels_model * mult,
                    d_model=d_model,
                    dropout=dropout,
                )
                layers.append(sp_trf)

            self.up_blocks.append(EmbeddingWrapper(*layers))

            down_ch = track_chans.pop()

            # and again, same dimension ...
            layers = []

            layers.append(ResBlock(ch_out + down_ch, d_emb=self.t_emb_dim, channels_out=ch_out))

            if i in attention_resolutions:
                layers.append(
                    SpatialTransformer(
                        in_channels=ch_out,
                        d_q=ch_out,
                        d_cross=context_dim if context_dim else ch_out,
                        d_model=d_model,
                        dropout=dropout,
                    )
                )

            # ... with an upsampling layer at all but the last, since at the last we are matching the initial convolutional layer and a res + spatial transformer block, not an upsampling layer
            if i > 0:
                layers.append(
                    ResBlock(ch_out, d_emb=self.t_emb_dim, channels_out=ch_out, resample="up")
                )

            self.up_blocks.append(EmbeddingWrapper(*layers))

        # output block that normalizes and maps back to
        self.out_block = nn.Sequential(
            nn.GroupNorm(32, channels_model),
            nn.SiLU(),
            nn.Conv2d(channels_model, channels_out, kernel_size=3, padding=1),
        )

    def forward(self, x, timesteps=None, context=None):
        """
        Args:
            x (torch.tensor): input tensor
            timesteps (torch.tensor): Size (B,) tensor containing time indices to be turned into embeddings.
            context (torch.tensor): context tensor
        """
        if context is None:
            assert self.context_dim is None, "Must pass context if context_dimension is set"
        else:
            assert (
                self.context_dim is not None
            ), "You must set context_dim when creating the model if planning on passing context embeddings."

        if timesteps is not None:
            timesteps = time_embeddings(timesteps, self.t_emb_dim)

        # downsample
        downsampled = []
        for block in self.down_blocks:
            x = block(x, timesteps, context)
            downsampled.append(x)

        # middle block
        x = self.middle_block(x, timesteps, context)

        # upsample
        for block in self.up_blocks:
            x = torch.cat([x, downsampled.pop()], dim=1)
            x = block(x, timesteps, context)

        x = self.out_block(x)
        return x
