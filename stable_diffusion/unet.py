import torch
import torch.nn as nn

from .resblocks import ResBlock
from .spatial_transformer import SpatialTransformer


class EmbeddingWrapper(nn.Sequential):
    """
    Wrapper for a sequence of layers that can handle embeddings
    """

    def forward(self, x, t_emb=None, context=None):
        """
        Args:
            x (torch.tensor): input tensor
            t_emb (torch.tensor): time embedding
            context (torch.tensor): context tensor
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
    def __init__(
        self,
        channels_in,
        channels_model,
        channels_out,
        t_emb_dim,
        context_dim,
        d_model,
        channel_mults=[1, 2, 4, 8],
        dropout=0.0
    ):
        super().__init__()

        self.channel_mults = channel_mults
        self.channels_in = channels_in
        self.channels_model = channels_model
        self.context_dim = context_dim

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

            resblock = ResBlock(channels_in=ch_in, d_emb=t_emb_dim, channels_out=ch_out)
            sp_transformer = SpatialTransformer(
                in_channels=ch_out,
                d_q=ch_out,
                d_cross=context_dim if context_dim else ch_out,
                d_model=d_model,
                dropout=dropout,
            )

            self.down_blocks.append(EmbeddingWrapper(resblock, sp_transformer))

            track_chans.append(ch_out)

            # downsample after every mult except the last
            if i != len(channel_mults) - 1:
                res_ds = ResBlock(ch_out, d_emb=t_emb_dim, channels_out=ch_out, resample="down")
                self.down_blocks.append(EmbeddingWrapper(res_ds))
                track_chans.append(ch_out)
                ch_in = ch_out

        # middle block, this is Res, Attention, Res
        # ch_out is the last channel dimension for constructing the downsampling layers
        self.middle_block = EmbeddingWrapper(
            ResBlock(channels_in=ch_out, d_emb=t_emb_dim),
            SpatialTransformer(
                in_channels=ch_out,
                d_q=ch_out,
                d_cross=context_dim if context_dim else ch_out,
                d_model=d_model,
                dropout=dropout,
            ),
            ResBlock(channels_in=ch_out, d_emb=t_emb_dim),
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
            res1 = ResBlock(ch_out + down_ch, d_emb=t_emb_dim, channels_out=channels_model * mult)

            # this block will output this many channels, we set it here since we want the next iteration to start the channels of the previous block.
            ch_out = channels_model * mult

            sp_trf = SpatialTransformer(
                in_channels=channels_model * mult,
                d_q=channels_model * mult,
                d_cross=context_dim if context_dim else channels_model * mult,
                d_model=d_model,
                dropout=dropout,
            )

            self.up_blocks.append(EmbeddingWrapper(res1, sp_trf))

            down_ch = track_chans.pop()

            # and again, same dimension ...
            layers = []

            layers.extend(
                [
                    ResBlock(ch_out + down_ch, d_emb=t_emb_dim, channels_out=ch_out),
                    SpatialTransformer(
                        in_channels=ch_out,
                        d_q=ch_out,
                        d_cross=context_dim if context_dim else ch_out,
                        d_model=d_model,
                        dropout=dropout,
                    ),
                ]
            )

            # ... with an upsampling layer at all but the last, since at the last we are matching the initial convolutional layer and a res + spatial transformer block, not an upsampling layer
            if i > 0:
                layers.append(
                    ResBlock(ch_out, d_emb=t_emb_dim, channels_out=ch_out, resample="up")
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
            timesteps (torch.tensor): time embedding
            context (torch.tensor): context tensor
        """
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
