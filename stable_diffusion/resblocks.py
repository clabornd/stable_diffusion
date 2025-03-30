import torch
import torch.nn as nn
import torch.nn.functional as F

# This resnet block needs:
# The input


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class Upsample(nn.Module):
    def __init__(self, channels_in, channels_out=None, dims=2, use_conv=False, padding=1):
        super().__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out or channels_in
        self.dims = dims

        if use_conv:
            self.conv = conv_nd(dims, channels_in, self.channels_out, 3, padding=padding)

    def forward(self, x):
        if self.dims == 3:
            _, _, h, w, d = x.size()
            # upsampling only occurs in the last two dimensions for some reason
            x = F.interpolate(x, (h, w * 2, d * 2), scale_factor=2, mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        if hasattr(self, "conv"):
            x = self.conv(x)

        return x


class DownSample(nn.Module):
    def __init__(self, channels_in, channels_out=None, dims=2, use_conv=False, padding=1):
        super().__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out or channels_in
        self.dims = dims

        if use_conv:
            self.downsample = conv_nd(
                dims, channels_in, self.channels_out, 3, stride=2, padding=padding
            )
        else:
            self.downsample = avg_pool_nd(dims, kernel_size=2, stride=2)

    def forward(self, x):
        return self.downsample(x)


class ResBlock(nn.Module):
    def __init__(self, channels_in, d_emb, channels_out=None, resample=None, dropout=0.0):
        super().__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out or channels_in

        self.group_norm_in = nn.GroupNorm(32, channels_in)

        self.conv_in = conv_nd(2, channels_in, self.channels_out, 3, padding=1)
        self.conv_out = conv_nd(2, self.channels_out, self.channels_out, 3, padding=1)

        if resample == "down":
            self.resample_h = DownSample(channels_in, dims=2, use_conv=False)
            self.resample_x = DownSample(channels_in, dims=2, use_conv=False)
        elif resample == "up":
            self.resample_h = Upsample(channels_in, dims=2, use_conv=False)
            self.resample_x = Upsample(channels_in, dims=2, use_conv=False)
        else:
            self.resample_x = self.resample_h = nn.Identity()

        if self.channels_out == channels_in:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(2, channels_in, self.channels_out, 1)

        self.time_emb = nn.Sequential(nn.SiLU(), nn.Linear(d_emb, self.channels_out))

        self.out_block = nn.Sequential(
            nn.GroupNorm(32, self.channels_out), nn.SiLU(), nn.Dropout(dropout), self.conv_out
        )

    def forward(self, x: torch.tensor, emb: torch.tensor):
        """
        Args:
            x (torch.tensor): Input tensor of shape (B, C, H, W)
            emb (torch.tensor): Time embedding of dimension (B, D)

        Returns:
            torch.tensor: Output tensor of shape (B, C*, H*, W*)
        """

        h = self.group_norm_in(x)
        h = F.silu(h)
        h = self.resample_h(h)
        h = self.conv_in(h)

        emb = self.time_emb(emb)
        h = h + emb[:, :, None, None]  # expand spatial dims, add along channel dim

        h = self.out_block(h)

        x = self.resample_x(x)

        return h + self.skip_connection(x)
