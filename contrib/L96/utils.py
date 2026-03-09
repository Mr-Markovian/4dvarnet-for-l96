import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import xarray as xr
import src.data
import xarray as xr
import os

def cosanneal_lr_adam_UNet(lit_mod, lr, T_max=100, weight_decay=0.):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.parameters(), "lr": lr},
        ], weight_decay=weight_decay 
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }

def get_constant_crop(patch_dims, crop, dim_order=("time", "lat", "lon")):
    """
    Returns a 0/1 crop mask with shape [time, lat, lon] (if lon exists)
    or [time, lat] (if lon is not in patch_dims).

    patch_dims: dict like {time:200, lat:40, lon:1}
    crop: dict like {time:10, lat:5, lon:0}
    """
    # Only keep dims that exist in patch_dims
    dim_order = [d for d in dim_order if d in patch_dims]

    patch_weight = np.zeros([patch_dims[d] for d in dim_order], dtype="float32")

    mask = tuple(
        slice(crop.get(d, 0), -crop.get(d, 0)) if crop.get(d, 0) > 0 else slice(None, None)
        for d in dim_order
    )

    patch_weight[mask] = 1.0
    return patch_weight



def get_triang_time_wei(patch_dims, offset=0, crop=None, dim_order=("time", "lat", "lon")):
    """
    Triangular weight along time, with optional cropping.
    Output is made compatible with your model: [1, time, lat].

    If lon exists, it is squeezed out.
    """
    crop = crop or {}

    pw = get_constant_crop(patch_dims, crop=crop, dim_order=dim_order)
    # pw shape: [time, lat, lon] or [time, lat]

    # Build triangular time ramp (shape [time, 1, 1] or [time, 1])
    T = patch_dims["time"]

    if pw.ndim == 3:
        # [time, lat, lon]
        tri = np.fromfunction(
            lambda t, y, x: (1 - np.abs(offset + 2 * t - T) / T),
            pw.shape,
            dtype=float
        ).astype(np.float32)

        w = tri * pw
        w = np.squeeze(w, axis=-1)   # drop lon -> [time, lat]

    elif pw.ndim == 2:
        # [time, lat]
        tri = np.fromfunction(
            lambda t, y: (1 - np.abs(offset + 2 * t - T) / T),
            pw.shape,
            dtype=float
        ).astype(np.float32)

        w = tri * pw

    else:
        raise ValueError(f"Unexpected pw.ndim={pw.ndim}, pw.shape={pw.shape}")

    # Add channel dim -> [1, time, lat]
    return w[None, ...]


def get_constant_time_wei(patch_dims, offset=0, crop=None, dim_order=("time", "lat", "lon")):
    """
    Returns a constant weighting mask (all ones in the non-cropped region, zeros elsewhere)
    with shape [1, time, lat], compatible with get_triang_time_wei.
    """
    crop = crop or {}

    pw = get_constant_crop(patch_dims, crop=crop, dim_order=dim_order)
    # pw shape: [time, lat, lon] or [time, lat]

    # Squeeze lon if present
    if pw.ndim == 3:
        w = np.squeeze(pw, axis=-1)   # drop lon -> [time, lat]
    elif pw.ndim == 2:
        w = pw
    else:
        raise ValueError(f"Unexpected pw.ndim={pw.ndim}, pw.shape={pw.shape}")

    # Add channel dim -> [1, time, lat]
    return w[None, ...]


def load_l96_data(path, obs_from_tgt=False):
    ds = (
        xr.open_dataset(path)
        .load()
        .assign(
            input=lambda ds: ds['obs'],
            tgt=lambda ds: ds['truth']
        )
    )
    
    return (
        ds[[*src.data.TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )


def load_l96_data_multi(paths):
    """Load multiple trajectory netCDFs, return list of DataArrays"""
    return [load_l96_data(p) for p in paths]

# For later if we keep adding trajectories and want to load them all at once:
# def load_l96_data_multi(path_pattern):
#     """Load all matching files e.g. '/data/L96_traj_*.nc'"""
#     import glob
#     paths = sorted(glob.glob(path_pattern))
#     print(f"Found {len(paths)} trajectories")
#     return [load_l96_data(p) for p in paths]

def load_l96_data_identity(path, obs_from_tgt=False):
    ds = (
        xr.open_dataset(path)
        .load()
        .assign(
            input=lambda ds: ds['truth'],
            tgt=lambda ds: ds['truth']
        )
    )
    
    return (
        ds[[*src.data.TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )    


# Unet parts 

""" Parts of the U-Net model
    -- modified from https://github.com/milesial/Pytorch-UNet """


class StandardBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 kernel_size=3, dilation=1, sf=None):
        super().__init__()
        padding = kernel_size // 2
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=kernel_size,
                padding=padding, bias=False, dilation=dilation),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=kernel_size,
                padding=padding, bias=False, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class StandardBlock_periodic(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 kernel_size=3, dilation=1, sf=None):
        super().__init__()
        padding = kernel_size // 2
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=kernel_size,
                padding=padding, padding_mode='circular', bias=False, dilation=dilation),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=kernel_size,
                padding=padding, padding_mode='circular', bias=False, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 kernel_size=3, sf=1):
        super().__init__()
        self._scaling_factor = sf
        
        padding = kernel_size // 2
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=kernel_size,
                padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=kernel_size,
                padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if in_channels != out_channels:
            self.projection_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

class ResBlock_periodic(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 kernel_size=3, sf=1):
        super().__init__()
        self._scaling_factor = sf
        
        padding = kernel_size // 2
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=kernel_size,
                padding=padding, padding_mode='circular', bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=kernel_size,
                padding=padding, padding_mode='circular', bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if in_channels != out_channels:
            self.projection_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding_mode='circular', bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.double_conv(x)

        if hasattr(self, 'projection_conv'):
            x = self.projection_conv(x)
            
        out = out * self._scaling_factor + x

        return F.relu(out)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, block, **kwargs):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            block(in_channels, out_channels, **kwargs)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Down_periodic(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, block, **kwargs):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            block(in_channels, out_channels, **kwargs)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, block=ResBlock, bilinear=True,
                 **kwargs):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = block(in_channels, out_channels, in_channels // 2,
                              **kwargs)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = block(in_channels, out_channels,
                              **kwargs)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up_periodic(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, block=ResBlock_periodic, bilinear=True,
                 **kwargs):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = block(in_channels, out_channels, in_channels // 2,
                              **kwargs)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = block(in_channels, out_channels,
                              **kwargs)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2], mode='circular')
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                             bias=False)

    def forward(self, x):
        return self.out(x)
    
class OutConv_periodic(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_periodic, self).__init__()
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                             bias=False, padding_mode='circular')

    def forward(self, x):
        return self.out(x)
        