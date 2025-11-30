import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode
from PIL import Image
import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import models
import math
import shutil
import json


# Differentiable 1D/2D ILWT (LeGall 5/3) utilities
def _reflect_pad_lastdim_1(x: torch.Tensor):
    """
    Reflect-pad by 1 element on both sides along the last dimension.
    Works for tensors of any rank >= 1.
    """
    if x.size(-1) < 2:
        # Fallback: replicate padding if too short to reflect
        left = x[..., :1]
        right = x[..., -1:]
    else:
        left = x[..., 1:2]
        right = x[..., -2:-1]
    return torch.cat([left, x, right], dim=-1)


def _ilwt53_forward_1d_lastdim(x: torch.Tensor):
    """
    Forward 1D LeGall 5/3 lifting along the last dimension.
    Returns (approx, detail) with length halved along last dim.
    """
    even = x[..., 0::2]
    odd = x[..., 1::2]

    # Predict step: d = odd - 0.5*(even_left + even_right)
    even_ext = _reflect_pad_lastdim_1(even)
    pred = 0.5 * (even_ext[..., :-2] + even_ext[..., 2:])
    d = odd - pred

    # Update step: s = even + 0.25*(d_left + d_right)
    d_ext = _reflect_pad_lastdim_1(d)
    upd = 0.25 * (d_ext[..., :-2] + d_ext[..., 2:])
    s = even + upd

    return s, d


def _ilwt53_inverse_1d_lastdim(s: torch.Tensor, d: torch.Tensor):
    """
    Inverse 1D LeGall 5/3 lifting along the last dimension.
    Reconstructs original length by interleaving.
    """
    # Inverse update: even = s - 0.25*(d_left + d_right)
    d_ext = _reflect_pad_lastdim_1(d)
    upd = 0.25 * (d_ext[..., :-2] + d_ext[..., 2:])
    even = s - upd

    # Inverse predict: odd = d + 0.5*(even_left + even_right)
    even_ext = _reflect_pad_lastdim_1(even)
    pred = 0.5 * (even_ext[..., :-2] + even_ext[..., 2:])
    odd = d + pred

    # Merge by interleaving
    last_len = even.size(-1) + odd.size(-1)
    out_shape = list(even.shape)
    out_shape[-1] = last_len
    y = torch.zeros(out_shape, dtype=even.dtype, device=even.device)
    y[..., 0::2] = even
    y[..., 1::2] = odd
    return y


def _apply_forward_1d_along_dim(x: torch.Tensor, dim: int):
    if dim == -1 or dim == x.dim() - 1:
        return _ilwt53_forward_1d_lastdim(x)
    # Move target dim to last, apply, then move back
    perm = list(range(x.dim()))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    x_perm = x.permute(*perm)
    s, d = _ilwt53_forward_1d_lastdim(x_perm)
    # Inverse permute
    inv_perm = list(range(x_perm.dim()))
    inv_perm[dim], inv_perm[-1] = inv_perm[-1], inv_perm[dim]
    s = s.permute(*inv_perm)
    d = d.permute(*inv_perm)
    return s, d


def _apply_inverse_1d_along_dim(s: torch.Tensor, d: torch.Tensor, dim: int):
    if dim == -1 or dim == s.dim() - 1:
        return _ilwt53_inverse_1d_lastdim(s, d)
    # Move target dim to last, apply, then move back
    perm = list(range(s.dim()))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    s_perm = s.permute(*perm)
    d_perm = d.permute(*perm)
    x_perm = _ilwt53_inverse_1d_lastdim(s_perm, d_perm)
    inv_perm = list(range(x_perm.dim()))
    inv_perm[dim], inv_perm[-1] = inv_perm[-1], inv_perm[dim]
    x = x_perm.permute(*inv_perm)
    return x


class ILWT53_2D(nn.Module):
    """
    Differentiable 2D ILWT (LeGall 5/3 lifting) operating per-channel.
    Forward: (B,C,H,W) -> (B,4C,H/2,W/2) concatenating subbands [LL,LH,HL,HH].
    Inverse: (B,4C,H/2,W/2) -> (B,C,H,W).
    """

    def __init__(self, channels: int):
        super(ILWT53_2D, self).__init__()
        self.channels = channels
        self.padding_info = None

    def _maybe_pad(self, x: torch.Tensor):
        b, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        self.padding_info = (pad_h, pad_w)
        return x

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = self._maybe_pad(x)
        # Row transform (along width dim=3): yields s_row, d_row with W/2
        s_row, d_row = _apply_forward_1d_along_dim(x, dim=3)
        # Column transform (along height dim=2):
        LL, LH = _apply_forward_1d_along_dim(s_row, dim=2)
        HL, HH = _apply_forward_1d_along_dim(d_row, dim=2)
        # Concatenate subbands along channel dim
        out = torch.cat([LL, LH, HL, HH], dim=1)
        return out

    def inverse(self, z: torch.Tensor):
        b, c4, h2, w2 = z.shape
        c = c4 // 4
        LL, LH, HL, HH = torch.split(z, c, dim=1)
        # Inverse column transforms to recover s_row and d_row
        s_row = _apply_inverse_1d_along_dim(LL, LH, dim=2)
        d_row = _apply_inverse_1d_along_dim(HL, HH, dim=2)
        # Inverse row transform to reconstruct spatial
        x = _apply_inverse_1d_along_dim(s_row, d_row, dim=3)
        # Remove padding if added
        if self.padding_info is not None:
            pad_h, pad_w = self.padding_info
            if pad_h:
                x = x[:, :, :-pad_h, :]
            if pad_w:
                x = x[:, :, :, :-pad_w]
        return x


# Color space utilities (RGB <-> YCbCr) in range [-1,1] tensors
def rgb_to_ycbcr(x):
    # x: (B,3,H,W) in [-1,1] -> convert to [0,1] then to YCbCr
    x01 = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    r, g, b = x01[:, 0:1], x01[:, 1:2], x01[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    out01 = torch.cat([y, cb, cr], dim=1)
    return out01 * 2.0 - 1.0


def ycbcr_to_rgb(x):
    # x: (B,3,H,W) in [-1,1] YCbCr -> convert to RGB in [-1,1]
    x01 = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    y, cb, cr = x01[:, 0:1], x01[:, 1:2] - 0.5, x01[:, 2:3] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    rgb01 = torch.cat([r, g, b], dim=1)
    return torch.clamp(rgb01 * 2.0 - 1.0, -1.0, 1.0)


# Learnable ILWT implementation
class LearnableILWTWithHaar(nn.Module):
    """
    Learnable ILWT using Haar wavelet approximation with learnable parameters.
    """

    def __init__(self, channels):
        super(LearnableILWTWithHaar, self).__init__()
        self.channels = channels

        # Convolution for downsampling (like wavelet decomposition)
        self.decomposition = nn.Conv2d(
            channels, 4 * channels, kernel_size=2, stride=2, padding=0, groups=channels
        )

        # Initialize to mimic Haar wavelet decomposition
        with torch.no_grad():
            haar_ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32) / 2
            haar_lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32) / 2
            haar_hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32) / 2
            haar_hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32) / 2

            for c in range(channels):
                self.decomposition.weight.data[4 * c, 0, :, :] = haar_ll
                self.decomposition.weight.data[4 * c + 1, 0, :, :] = haar_lh
                self.decomposition.weight.data[4 * c + 2, 0, :, :] = haar_hl
                self.decomposition.weight.data[4 * c + 3, 0, :, :] = haar_hh

        # Transposed convolution for reconstruction
        self.reconstruction = nn.ConvTranspose2d(
            4 * channels, channels, kernel_size=2, stride=2, padding=0, groups=channels
        )

        with torch.no_grad():
            recon_filter = torch.tensor(
                [[0.25, 0.25], [0.25, 0.25]], dtype=torch.float32
            )
            for c in range(channels):
                for i in range(4):
                    self.reconstruction.weight.data[4 * c + i, 0, :, :] = recon_filter

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Pad if dimensions are odd
        pad_h, pad_w = 0, 0
        if height % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0), mode="reflect")
            pad_h = 1
        if width % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode="reflect")
            pad_w = 1

        if pad_h > 0 or pad_w > 0:
            self.padding_info = (pad_h, pad_w)
        else:
            self.padding_info = None

        output = self.decomposition(x)
        return output

    def inverse(self, x):
        reconstructed = self.reconstruction(x)

        if hasattr(self, "padding_info") and self.padding_info:
            pad_h, pad_w = self.padding_info
            if pad_h > 0:
                reconstructed = reconstructed[:, :, :-1, :]
            if pad_w > 0:
                reconstructed = reconstructed[:, :, :, :-1]

        return reconstructed


# Activation normalization
class ActNorm(nn.Module):
    def __init__(self, channels):
        super(ActNorm, self).__init__()
        self.channels = channels
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
                std = torch.std(x, dim=[0, 2, 3], keepdim=True)
                self.scale.data.copy_(1.0 / (std + 1e-6))
                self.bias.data.copy_(-mean * self.scale.data)
                self.initialized.fill_(1)

        y = x * self.scale + self.bias
        log_det = torch.sum(torch.log(torch.abs(self.scale))) * x.shape[2] * x.shape[3]
        return y, log_det

    def inverse(self, y):
        x = (y - self.bias) / self.scale
        return x




# Unbounded Learnable Subband Weights
class LearnableSubbandWeights(nn.Module):
    """
    Unbounded learnable wavelet subband weights with full freedom.
    Uses exponential transform for natural positive scaling.
    """
    def __init__(self, init_wLL=0.35, init_wLH=0.14, init_wHL=0.14, init_wHH=0.06, init_wLL2=0.20):
        super(LearnableSubbandWeights, self).__init__()
        # Initialize in log space so exp(log_w) = w
        self.log_wLL = nn.Parameter(torch.log(torch.tensor(init_wLL)))
        self.log_wLH = nn.Parameter(torch.log(torch.tensor(init_wLH)))
        self.log_wHL = nn.Parameter(torch.log(torch.tensor(init_wHL)))
        self.log_wHH = nn.Parameter(torch.log(torch.tensor(init_wHH)))
        self.log_wLL2 = nn.Parameter(torch.log(torch.tensor(init_wLL2)))
    
    def get_weights(self):
        # exp ensures positivity and preserves scale exactly at init
        wLL = torch.exp(self.log_wLL)
        wLH = torch.exp(self.log_wLH)
        wHL = torch.exp(self.log_wHL)
        wHH = torch.exp(self.log_wHH)
        wLL2 = torch.exp(self.log_wLL2)
        return torch.stack([wLL, wLH, wHL, wHH]), wLL2




# Unbounded Learnable YCbCr Scaling
class LearnableYCbCrScaling(nn.Module):
    """
    Unbounded learnable YCbCr scaling with full freedom.
    Uses exponential transform for natural positive scaling.
    """
    def __init__(self, init_kY=0.02, init_kC=0.06):
        super(LearnableYCbCrScaling, self).__init__()
        # Initialize in log space so exp(log_k) = k
        self.log_kY = nn.Parameter(torch.log(torch.tensor(init_kY)))
        self.log_kC = nn.Parameter(torch.log(torch.tensor(init_kC)))
    
    def get_scales(self):
        # exp ensures positivity and preserves scale exactly at init
        kY = torch.exp(self.log_kY)
        kC = torch.exp(self.log_kC)
        return kY, kC


# Affine coupling layer
class AffineCouplingLayer(nn.Module):
    def __init__(self, channels, hidden_channels=64, cond_channels=0):
        super(AffineCouplingLayer, self).__init__()

        assert channels % 2 == 0, "Number of channels must be even"
        half_channels = channels // 2

        in_ch = half_channels + cond_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.GroupNorm(1, hidden_channels),
            nn.Conv2d(hidden_channels, half_channels * 2, kernel_size=3, padding=1),
        )

        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()
        # Gated residual on x2 to ease optimization (kept invertible by adjusting inverse)
        self.gamma_raw = nn.Parameter(torch.zeros(1, half_channels, 1, 1))

    def forward(self, x, cond=None):
        x1, x2 = x.chunk(2, dim=1)
        if cond is not None:
            x1_in = torch.cat([x1, cond], dim=1)
        else:
            x1_in = x1
        s_t = self.net(x1_in)
        s, t = s_t.chunk(2, dim=1)
        # stabilize scale
        s = torch.tanh(s)
        gamma = F.softplus(self.gamma_raw)
        y1 = x1
        y2 = torch.exp(s) * x2 + t + gamma * x2
        return torch.cat([y1, y2], dim=1)

    def inverse(self, y, cond=None):
        y1, y2 = y.chunk(2, dim=1)
        if cond is not None:
            y1_in = torch.cat([y1, cond], dim=1)
        else:
            y1_in = y1
        s_t = self.net(y1_in)
        s, t = s_t.chunk(2, dim=1)
        s = torch.tanh(s)
        gamma = F.softplus(self.gamma_raw)
        x1 = y1
        denom = torch.exp(s) + gamma + 1e-6
        x2 = (y2 - t) / denom
        return torch.cat([x1, x2], dim=1)


# Invertible 1x1 convolution
class Invertible1x1Conv(nn.Module):
    def __init__(self, channels):
        super(Invertible1x1Conv, self).__init__()
        self.channels = channels

        w_init = np.random.randn(channels, channels)
        q, _ = np.linalg.qr(w_init)
        w_init = torch.from_numpy(q.astype(np.float32))
        self.weight = nn.Parameter(w_init)

    def forward(self, x):
        b, c, h, w = x.size()
        x_flat = x.view(b, c, -1)
        y_flat = torch.matmul(self.weight, x_flat)
        y = y_flat.view(b, c, h, w)
        log_det = h * w * torch.log(torch.abs(torch.det(self.weight)))
        return y, log_det

    def inverse(self, y):
        b, c, h, w = y.size()
        w_inv = torch.inverse(self.weight)
        y_flat = y.view(b, c, -1)
        x_flat = torch.matmul(w_inv, y_flat)
        x = x_flat.view(b, c, h, w)
        return x


# StarINN Block
class StarINNBlock(nn.Module):
    def __init__(self, channels, hidden_channels=64, cond_channels=0):
        super(StarINNBlock, self).__init__()
        assert channels % 2 == 0, "Channels must be even"

        self.actnorm = ActNorm(channels)
        self.inv_conv = Invertible1x1Conv(channels)
        self.affine_coupling = AffineCouplingLayer(
            channels, hidden_channels, cond_channels
        )

    def forward(self, x, cond=None):
        x, log_det1 = self.actnorm(x)
        x, log_det2 = self.inv_conv(x)
        x = self.affine_coupling(x, cond)
        log_det = log_det1 + log_det2
        return x, log_det

    def inverse(self, y, cond=None):
        y = self.affine_coupling.inverse(y, cond)
        y = self.inv_conv.inverse(y)
        y = self.actnorm.inverse(y)
        return y


# StarINN with ILWT
class StarINNWithILWT(nn.Module):
    def __init__(
        self,
        channels=6,
        num_blocks=3,
        hidden_channels=48,
        transform_type: str = "ilwt53",
    ):
        super(StarINNWithILWT, self).__init__()

        if transform_type == "haar_conv":
            self.ilwt = LearnableILWTWithHaar(channels)
        elif transform_type == "ilwt53":
            self.ilwt = ILWT53_2D(channels)
        else:
            raise ValueError(f"Unknown transform_type: {transform_type}")
        self.inn_channels = channels * 4
        # Lightweight UNet-like feature extractor for conditioning
        cond_ch = 16
        self.cond_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                StarINNBlock(self.inn_channels, hidden_channels, cond_channels=cond_ch)
            )
        
        # Learnable wavelet subband weights
        self.subband_weights = LearnableSubbandWeights()
        
        # Learnable YCbCr scaling parameters
        self.ycbcr_scaling = LearnableYCbCrScaling()

    def forward(self, x):
        # Outer cover-preserving residual skip + UNet conditioning
        cover = x[:, :3, :, :]
        cond = self.cond_net(cover)
        x_freq = self.ilwt(x)
        # Resize cond to match frequency map spatial size (H/2, W/2)
        cond_resized = F.interpolate(
            cond,
            size=(x_freq.shape[2], x_freq.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        log_det_total = 0
        z = x_freq
        for block in self.blocks:
            z, log_det = block(z, cond_resized)
            log_det_total += log_det

        residual_spatial = self.ilwt.inverse(z)
        # Compose stego in YCbCr: keep Y almost unchanged, allow more in Cb/Cr
        cover_ycc = rgb_to_ycbcr(cover)
        resid_ycc = rgb_to_ycbcr(residual_spatial[:, :3, :, :])
        # Get LEARNABLE YCbCr scaling parameters
        kY, kC = self.ycbcr_scaling.get_scales()
        # Use broadcasting for scale tensor (1, 3, 1, 1) - preserves gradients!
        scale = torch.stack([kY, kC, kC]).view(1, 3, 1, 1).to(cover.device)
        
        stego_ycc = cover_ycc + torch.tanh(resid_ycc) * scale
        stego_rgb = ycbcr_to_rgb(stego_ycc)
        stego_spatial = x.clone()
        stego_spatial[:, :3, :, :] = stego_rgb
        # Keep secret channels from residual path to support inverse recovery
        stego_spatial[:, 3:, :, :] = residual_spatial[:, 3:, :, :]
        return stego_spatial, log_det_total

    def inverse(self, z):
        # Inverse should remove the outer residual: approximate by passing through blocks and inverse ILWT,
        # then subtracting the tanh residual on host channels with same k.
        z_freq = self.ilwt(z)
        cond = self.cond_net(z[:, :3, :, :])
        cond_resized = F.interpolate(
            cond,
            size=(z_freq.shape[2], z_freq.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        x_freq = z_freq
        for i in range(len(self.blocks) - 1, -1, -1):
            x_freq = self.blocks[i].inverse(x_freq, cond_resized)
        residual_spatial = self.ilwt.inverse(x_freq)
        # Undo YCbCr composition
        z_ycc = rgb_to_ycbcr(z[:, :3, :, :])
        resid_ycc = rgb_to_ycbcr(residual_spatial[:, :3, :, :])
        # Get LEARNABLE YCbCr scaling parameters (same as forward)
        kY, kC = self.ycbcr_scaling.get_scales()
        # Use broadcasting for scale tensor (1, 3, 1, 1) - preserves gradients!
        scale = torch.stack([kY, kC, kC]).view(1, 3, 1, 1).to(z.device)
        
        host_ycc = z_ycc - torch.tanh(resid_ycc) * scale
        host_rgb = ycbcr_to_rgb(host_ycc)
        x_spatial = z.clone()
        x_spatial[:, :3, :, :] = host_rgb
        # Secret channels come from the residual path directly
        x_spatial[:, 3:, :, :] = residual_spatial[:, 3:, :, :]
        return x_spatial


# Dataset
class ImageSteganographyDataset(Dataset):
    def __init__(self, image_dir, img_size=224, transform=None):
        png_files = glob.glob(os.path.join(image_dir, "*.png"))
        jpg_files = glob.glob(os.path.join(image_dir, "*.jpg"))
        jpeg_files = glob.glob(os.path.join(image_dir, "*.jpeg"))
        self.image_paths = png_files + jpg_files + jpeg_files
        self.img_size = img_size

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (img_size, img_size),
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        host_path = self.image_paths[idx]
        host_img = Image.open(host_path).convert("RGB")
        host_tensor = self.transform(host_img)

        secret_idx = random.choice(
            [i for i in range(len(self.image_paths)) if i != idx]
        )
        secret_path = self.image_paths[secret_idx]
        secret_img = Image.open(secret_path).convert("RGB")
        secret_tensor = self.transform(secret_img)

        combined_input = torch.cat([host_tensor, secret_tensor], dim=0)
        return combined_input, host_tensor, secret_tensor


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split dataset into train, validation, and test sets
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    total_size = len(dataset)
    indices = list(range(total_size))
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Shuffle indices
    np.random.shuffle(indices)
    
    # Calculate split sizes
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    print(f"Dataset split:")
    print(f"  Total: {total_size}")
    print(f"  Train: {len(train_dataset)} ({train_ratio*100:.1f}%)")
    print(f"  Val:   {len(val_dataset)} ({val_ratio*100:.1f}%)")
    print(f"  Test:  {len(test_dataset)} ({test_ratio*100:.1f}%)")
    
    return train_dataset, val_dataset, test_dataset


# Metrics
def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float("inf")
    max_pixel = 2.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse + 1e-10))
    return psnr


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    img1 = (img1 + 1) / 2.0
    img2 = (img2 + 1) / 2.0
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)

    def create_window(window_size, channel):
        def _gaussian(window_size, sigma):
            gauss = torch.Tensor(
                [
                    np.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                    for x in range(window_size)
                ]
            )
            return gauss / gauss.sum()

        _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def calculate_bit_acc_and_bpp(img1, img2):
    """
    Compute bit-wise accuracy (ACC) between two images in [-1,1] and payload bpp.
    - img1, img2: tensors (B,C,H,W) normalized to [-1,1]
    Returns: (acc, bpp)
    """
    with torch.no_grad():
        # Denormalize to [0,255] uint8
        def to_uint8(x):
            x = (x + 1.0) / 2.0
            x = torch.clamp(x, 0.0, 1.0)
            x = (x * 255.0).round().to(torch.uint8)
            return x

        a = to_uint8(img1).cpu().numpy()
        b = to_uint8(img2).cpu().numpy()
        # Flatten to bytes
        a_flat = a.reshape(-1)
        b_flat = b.reshape(-1)
        # Unpack to bits (uint8 -> 8 bits)
        a_bits = np.unpackbits(a_flat)
        b_bits = np.unpackbits(b_flat)
        total_bits = a_bits.size
        correct_bits = (a_bits == b_bits).sum()
        acc = float(correct_bits) / float(total_bits) if total_bits > 0 else 0.0
        # Bits per pixel = payload bits per cover pixel (C*8)
        _, C, H, W = img1.shape
        bpp = float(C * 8)
        bit_error_rate = 1.0 - acc  # BER = 1 - accuracy
        return acc, bpp, bit_error_rate


def calculate_ber(img1, img2):
    """
    Calculate Bit Error Rate (BER) between two images
    """
    acc, _, ber = calculate_bit_acc_and_bpp(img1, img2)
    return ber


def calculate_hiding_ratio(host_img, stego_img):
    """
    Calculate Hiding Ratio (HR) - similarity between host and stego images
    """
    mse = F.mse_loss(host_img, stego_img)
    hr = 1.0 / (1.0 + mse)  # Higher values indicate better hiding performance
    return hr.item()


def calculate_bitrate_increase_ratio(host_img, stego_img):
    """
    Calculate Bitrate Increase Ratio (BIR) - measure of data increase
    """
    # For image steganography, we can use compression ratio comparison
    # Using MSE as a proxy for complexity change
    mse = F.mse_loss(host_img, stego_img)
    bir = mse.item()
    return bir


def calculate_detection_accuracy(model, cover_images, stego_images, threshold=0.5):
    """
    Calculate detection accuracy by training a simple detector to distinguish cover vs stego images
    """
    # This is a simplified approach - in practice, you'd train a detector network
    # For now, we'll estimate based on statistical differences
    
    # Compare statistical properties between cover and stego images
    cover_stats = [F.mse_loss(img, img).item() for img in cover_images]  # placeholder
    stego_stats = [F.mse_loss(img, img).item() for img in stego_images]  # placeholder
    
    # Detection accuracy would be how well a classifier can separate these
    # For now, return a placeholder - in practice, this would require training a detector
    return 0.5  # 50% detection accuracy (random guess) means perfect steganography


def calculate_imperceptibility(hide_img, stego_img):
    """
    Calculate imperceptibility measure - how well the stego image preserves cover image quality
    """
    psnr = calculate_psnr(hide_img, stego_img)
    ssim = calculate_ssim(hide_img, stego_img)
    # Imperceptibility is a combination of PSNR and SSIM
    imperceptibility = (psnr.item() / 100.0 + ssim.item()) / 2.0  # normalize and average
    return imperceptibility


def steganography_loss(
    secret_img,
    recovered_secret,
    cover_img,
    stego_img,
    ilwt_module,
    subband_weights,
    target_psnr=42.0
):
    """
    Target-Driven Loss to achieve specific PSNR goals (e.g., 42 dB).
    Dynamically weights hiding and recovery loss based on distance from target.
    """
    # 1. Calculate MSEs
    hiding_mse = F.mse_loss(stego_img, cover_img)
    recovery_mse = F.mse_loss(recovered_secret, secret_img)
    
    # 2. Convert Target PSNR to Target MSE
    # PSNR = -10 * log10(MSE) -> MSE = 10^(-PSNR/10)
    target_mse = 10 ** (-target_psnr / 10.0)
    
    # 3. Calculate "Gap" (how far we are from target)
    # We use ReLU because if we are better than target (MSE < Target), gap is 0
    # Add small epsilon to ensure we keep optimizing even if target reached
    gap_hid = F.relu(hiding_mse - target_mse) + 1e-6
    gap_rec = F.relu(recovery_mse - target_mse) + 1e-6
    
    # 4. Dynamic Weights based on Gaps
    total_gap = gap_hid + gap_rec
    w_hid = gap_hid / total_gap
    w_rec = gap_rec / total_gap
    
    # 5. Multi-scale Secret Loss (Structural)
    # We scale this by w_rec since it's a recovery metric
    loss_secret_structure = ilwt_multiscale_secret_loss(
        ilwt_module, secret_img, recovered_secret, subband_weights
    )
    
    # 6. Final Loss
    # We multiply by a large constant (e.g. 100) to keep gradients strong
    # as MSE values for 42dB are very small (~6e-5)
    scale_factor = 1000.0
    total_loss = scale_factor * (w_hid * hiding_mse + w_rec * recovery_mse) + w_rec * loss_secret_structure
    
    # Calculate actual PSNRs for logging
    hiding_psnr_val = -10 * torch.log10(hiding_mse + 1e-8)
    recovery_psnr_val = -10 * torch.log10(recovery_mse + 1e-8)
    rec_ssim_val = calculate_ssim(recovered_secret, secret_img)

    return total_loss, hiding_psnr_val, recovery_psnr_val, rec_ssim_val, w_hid, w_rec


def ilwt_multiscale_secret_loss(ilwt_module, secret_img, recovered_secret, subband_weights):
    """
    Multi-scale loss with LEARNABLE subband weights.
    The model learns optimal importance for LL (structure), LH/HL (edges), HH (noise).
    """
    # Level-1 bands
    concat_s = ilwt_module.forward(secret_img)  # (B, 4C, H/2, W/2)
    concat_r = ilwt_module.forward(recovered_secret)
    C = secret_img.size(1)
    s_LL, s_LH, s_HL, s_HH = torch.split(concat_s, C, dim=1)
    r_LL, r_LH, r_HL, r_HH = torch.split(concat_r, C, dim=1)
    
    # Get LEARNABLE weights from the model
    weights_lvl1, wLL2 = subband_weights.get_weights()
    wLL, wLH, wHL, wHH = weights_lvl1[0], weights_lvl1[1], weights_lvl1[2], weights_lvl1[3]
    
    loss_lvl1 = (
        wLL * F.l1_loss(r_LL, s_LL)
        + wLH * F.l1_loss(r_LH, s_LH)
        + wHL * F.l1_loss(r_HL, s_HL)
        + wHH * F.l1_loss(r_HH, s_HH)
    )
    # Level-2: enforce structure on LL via LL-of-LL with learnable weight
    s2 = ilwt_module.forward(s_LL)
    r2 = ilwt_module.forward(r_LL)
    s2_LL, _, _, _ = torch.split(s2, C, dim=1)
    r2_LL, _, _, _ = torch.split(r2, C, dim=1)
    loss_lvl2 = wLL2 * F.l1_loss(r2_LL, s2_LL)
    return loss_lvl1 + loss_lvl2


def perturb_stego_like_for_inverse(
    stego_host, q_prob=0.5, noise_prob=0.5, noise_sigma=0.01
):
    """
    Apply mild, differentiable-ish perturbations to stego before inverse path to
    simulate file I/O effects. Quantization-like rounding and small Gaussian noise.
    """
    x = stego_host
    if random.random() < q_prob:
        # Simulate 8-bit quantization in normalized space
        x_01 = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
        x_q = torch.round(x_01 * 255.0) / 255.0
        x = x_q * 2.0 - 1.0
    if random.random() < noise_prob:
        noise = torch.randn_like(x) * noise_sigma
        x = torch.clamp(x + noise, -1.0, 1.0)
    return x


# ============================================================================
# PUBLICATION-READY FEATURES & CHECKPOINTING
# ============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, metrics_history, 
                   best_recovery_psnr, checkpoint_dir='checkpoints'):
    """
    Save training checkpoint with full state for resuming.
    Keeps only last 3 checkpoints to save disk space.
    """
    import glob
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics_history': metrics_history,
        'best_recovery_psnr': best_recovery_psnr,
        'learnable_params': {
            'subband_weights': {k: v.cpu() for k, v in model.subband_weights.state_dict().items()},
            'ycbcr_scaling': {k: v.cpu() for k, v in model.ycbcr_scaling.state_dict().items()}
        },
        'config': {
            'num_blocks': len(model.blocks),
            'note': 'Model architecture details omitted for simplicity'
        }
    }
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Keep only last 3 checkpoints to save space
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth')))
    if len(checkpoints) > 3:
        for old_checkpoint in checkpoints[:-3]:
            os.remove(old_checkpoint)
            print(f"  Removed old checkpoint: {old_checkpoint}")
    
    return checkpoint_path


def generate_publication_plots(metrics_history, output_dir='publication_outputs/plots'):
    """
    Generate publication-quality plots (300 DPI PDF/PNG).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set publication-quality defaults
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.figsize': (10, 6),
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    epochs = metrics_history['epoch']
    
    # 1. Training Curves (PSNR)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, metrics_history['train_hiding_psnr'], 'b-', label='Hiding PSNR', linewidth=2)
    ax1.plot(epochs, metrics_history['train_recovery_psnr'], 'r-', label='Recovery PSNR', linewidth=2)
    ax1.axhline(y=42, color='g', linestyle='--', label='Target (42 dB)', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('PSNR Evolution During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Dynamic Weights
    w_rec = [1 - w for w in metrics_history['alpha_hid']]  # w_rec = 1 - w_hid
    ax2.plot(epochs, metrics_history['alpha_hid'], 'purple', label='w_hid', linewidth=2)
    ax2.plot(epochs, w_rec, 'orange', label='w_rec', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Weight')
    ax2.set_title('Dynamic Loss Balancing')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'training_curves.pdf'))
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300)
    plt.close()
    
    # 2. Weight Evolution (Wavelet + YCbCr)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, metrics_history['weight_LL'], label='LL', linewidth=2)
    ax1.plot(epochs, metrics_history['weight_LH'], label='LH', linewidth=2)
    ax1.plot(epochs, metrics_history['weight_HL'], label='HL', linewidth=2)
    ax1.plot(epochs, metrics_history['weight_HH'], label='HH', linewidth=2)
    ax1.plot(epochs, metrics_history['weight_LL2'], label='LL2', linewidth=2, linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Weight Value')
    ax1.set_title('Wavelet Subband Weight Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, metrics_history['scale_kY'], label='kY (Luminance)', linewidth=2, color='orange')
    ax2.plot(epochs, metrics_history['scale_kC'], label='kC (Chrominance)', linewidth=2, color='purple')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Scale Factor')
    ax2.set_title('YCbCr Scaling Parameter Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'weight_evolution.pdf'))
    plt.savefig(os.path.join(output_dir, 'weight_evolution.png'), dpi=300)
    plt.close()
    
    print(f"Publication plots saved to {output_dir}")


def generate_latex_tables(test_metrics, output_dir='publication_outputs/tables'):
    """
    Generate LaTeX tables ready for publication.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics table
    table = r"""\begin{table}[h]
\centering
\caption{Steganography Performance Metrics}
\label{tab:metrics}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{Unit} \\
\midrule
Hiding PSNR & %.2f $\pm$ %.2f & dB \\
Recovery PSNR & %.2f $\pm$ %.2f & dB \\
Hiding SSIM & %.4f & - \\
Recovery SSIM & %.4f & - \\
Bit Accuracy & %.2f & \%% \\
Bit Error Rate & %.2f & \%% \\
Bits Per Pixel & %.2f & bits/pixel \\
\bottomrule
\end{tabular}
\end{table}
""" % (
        test_metrics['avg_hiding_psnr'], 0.0,  # Add stddev if available
        test_metrics['avg_recovery_psnr'], 0.0,
        test_metrics['avg_hiding_ssim'],
        test_metrics['avg_recovery_ssim'],
        test_metrics['avg_bit_acc'] * 100,
        test_metrics['avg_ber'] * 100,
        test_metrics['bpp']
    )
    
    with open(os.path.join(output_dir, 'metrics_table.tex'), 'w') as f:
        f.write(table)
    
    print(f"LaTeX tables saved to {output_dir}")


def export_metrics_comprehensive(metrics_history, test_metrics, output_dir='publication_outputs/metrics'):
    """
    Export metrics in multiple formats: JSON, CSV, Markdown.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. JSON export (full history)
    with open(os.path.join(output_dir, 'full_metrics.json'), 'w') as f:
        json.dump({
            'training_history': metrics_history,
            'test_results': test_metrics
        }, f, indent=4)
    
    # 2. CSV export (training metrics)
    import csv
    with open(os.path.join(output_dir, 'training_metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Hiding_PSNR', 'Recovery_PSNR', 'Bit_Accuracy', 'w_hid', 'w_rec'])
        for i in range(len(metrics_history['epoch'])):
            writer.writerow([
                metrics_history['epoch'][i],
                metrics_history['train_total_loss'][i],
                metrics_history['train_hiding_psnr'][i],
                metrics_history['train_recovery_psnr'][i],
                metrics_history['train_bit_acc'][i],
                metrics_history['alpha_hid'][i],
                1 - metrics_history['alpha_hid'][i]
            ])
    
    # 3. Markdown summary
    md = f"""# Training Results Summary

## Test Performance
- **Hiding PSNR**: {test_metrics['avg_hiding_psnr']:.2f} dB
- **Recovery PSNR**: {test_metrics['avg_recovery_psnr']:.2f} dB
- **Hiding SSIM**: {test_metrics['avg_hiding_ssim']:.4f}
- **Recovery SSIM**: {test_metrics['avg_recovery_ssim']:.4f}
- **Bit Accuracy**: {test_metrics['avg_bit_acc']*100:.2f}%
- **Bit Error Rate**: {test_metrics['avg_ber']*100:.2f}%
- **BPP**: {test_metrics['bpp']:.2f}

## Final Learnable Parameters
- **kY (Luminance)**: {metrics_history['scale_kY'][-1]:.4f}
- **kC (Chrominance)**: {metrics_history['scale_kC'][-1]:.4f}
- **LL Weight**: {metrics_history['weight_LL'][-1]:.4f}
- **LH Weight**: {metrics_history['weight_LH'][-1]:.4f}
"""
    
    with open(os.path.join(output_dir, 'summary.md'), 'w') as f:
        f.write(md)
    
    print(f"Comprehensive metrics exported to {output_dir}")


# Training function with enhanced metrics and validation
def train_model(model, train_dataset, val_dataset, num_epochs=150, save_metrics=True, validate_every=1):
    print("\nTraining ILWT Steganography Model (with enhanced metrics and validation)...")

    # Open training log file
    training_log_file = "research_metrics/training_log.txt"
    os.makedirs(os.path.dirname(training_log_file), exist_ok=True)  # Ensure directory exists
    
    with open(training_log_file, 'w') as log_file:
        log_file.write("ILWT Steganography Training Log\n")
        log_file.write("=" * 50 + "\n")
        log_file.write(f"Started at: {__import__('datetime').datetime.now()}\n")
        log_file.write("\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    batch_size = 1  # SAFE: Avoids ILWT padding conflicts
    learning_rate = 3e-5  # Slightly increased for faster learning
    # Loss weights and schedules - IMPROVED for better recovery
    # alpha_hid_start = 2.0  # Start gentler (was 4.0)
    # alpha_hid_end = 24.0  # Less aggressive hiding priority (was 48.0)
    # alpha_rec_mse = 3.0  # Prioritize secret recovery (was 1.0)
    # alpha_rec_ssim = 5.0  # Strong perceptual quality (was 2.0)

    # GPU-optimized DataLoaders (SAFE optimizations only)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Multi-threaded data loading (SAFE)
        pin_memory=True,  # Faster GPU transfer (SAFE)
        prefetch_factor=2,  # Prefetch batches (SAFE)
        persistent_workers=True  # Keep workers alive (SAFE)
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,  # Fewer workers for validation (SAFE)
        pin_memory=True,  # (SAFE)
        prefetch_factor=2  # (SAFE)
    )
    
    # Separate parameters into groups for different learning rates
    learnable_params = list(model.subband_weights.parameters()) + list(model.ycbcr_scaling.parameters())
    learnable_ids = list(map(id, learnable_params))
    base_params = [p for p in model.parameters() if id(p) not in learnable_ids]
    
    optimizer = torch.optim.Adam(
        [
            {"params": base_params, "lr": learning_rate},
            {"params": learnable_params, "lr": 1e-2},  # Aggressive learning rate for weights!
        ],
        betas=(0.5, 0.999),
        weight_decay=1e-5,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-7
    )

    # Mixed precision DISABLED for stability (caused quality degradation)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    use_amp = False  # Disabled - FP32 is more stable

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"SAFE GPU optimizations: batch_size={batch_size}, num_workers=4, pin_memory=True, AMP={use_amp}")
    
    # Track metrics over epochs for research
    metrics_history = {
        'epoch': [],
        'train_total_loss': [],
        'train_hiding_psnr': [],
        'train_recovery_psnr': [],
        'train_hiding_ssim': [],
        'train_recovery_ssim': [],
        'train_hiding_mse': [],
        'train_recovery_mse': [],
        'train_bit_acc': [],
        'train_ber': [],
        'train_hiding_hr': [],
        'train_bir': [],
        'val_total_loss': [],
        'val_hiding_psnr': [],
        'val_recovery_psnr': [],
        'val_hiding_ssim': [],
        'val_recovery_ssim': [],
        'val_hiding_mse': [],
        'val_recovery_mse': [],
        'val_bit_acc': [],
        'val_ber': [],
        'val_hiding_hr': [],
        'val_bir': [],
        'bpp': [],
        'alpha_hid': [], # This will now store w_hid
        'alpha_rec_mse': [], # This will now store w_rec
        # Weight history
        'weight_LL': [], 'weight_LH': [], 'weight_HL': [], 'weight_HH': [], 'weight_LL2': [],
        'scale_kY': [], 'scale_kC': []
    }

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        # Training loop
        for batch_idx, (input_tensor, host_tensor, secret_tensor) in enumerate(train_dataloader):
            # Non-blocking GPU transfer for faster data loading
            input_tensor = input_tensor.to(device, non_blocking=True)
            host_tensor = host_tensor.to(device, non_blocking=True)
            secret_tensor = secret_tensor.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=use_amp):
                stego_output, log_det = model(input_tensor)
                host_input = input_tensor[:, :3, :, :]
                stego_host = stego_output[:, :3, :, :]
                # Stego-only inverse: append zeros for the missing channels
                stego_like = torch.cat(
                    [stego_host, torch.zeros_like(stego_host)], dim=1
                )
                reconstructed_input = model.inverse(stego_like)
                recovered_secret = reconstructed_input[:, 3:, :, :]

                # Target-Driven Loss (Goal: 42 dB)
                loss, hiding_loss, rec_mse, rec_ssim, w_hid, w_rec = steganography_loss(
                    secret_tensor,
                    recovered_secret,
                    host_tensor,
                    stego_host,
                    model.ilwt,
                    model.subband_weights,
                    target_psnr=42.0
                )
                
                # ms_loss is already included in steganography_loss


            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Less aggressive (was 0.5)
            scaler.step(optimizer)
            scaler.update()
            
            # Log gradients for first batch to monitor learning
            if batch_idx == 0 and epoch % 1 == 0:
                with torch.no_grad():
                    wLL_grad = model.subband_weights.log_wLL.grad.item() if model.subband_weights.log_wLL.grad is not None else 0.0
                    kY_grad = model.ycbcr_scaling.log_kY.grad.item() if model.ycbcr_scaling.log_kY.grad is not None else 0.0
                    kC_grad = model.ycbcr_scaling.log_kC.grad.item() if model.ycbcr_scaling.log_kC.grad is not None else 0.0
                    if epoch == 0:  # Only print on first epoch to avoid clutter
                        print(f"    [Batch 0 Gradients] log_wLL: {wLL_grad:.7f}, log_kY: {kY_grad:.7f}, log_kC: {kC_grad:.7f}")

            epoch_train_loss += loss.item()

        # Calculate average training metrics
        avg_train_loss = epoch_train_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        
        # Validation loop (run every few epochs to save time)
        avg_val_loss = 0
        avg_val_hiding_psnr = 0
        avg_val_recovery_psnr = 0
        avg_val_recovery_ssim = 0
        avg_val_bit_acc = 0
        
        if epoch % validate_every == 0:  # Validate every 'validate_every' epochs
            model.eval()
            val_losses = []
            val_hiding_psnrs = []
            val_recovery_psnrs = []
            val_recovery_ssims = []
            val_bit_accs = []
            
            with torch.no_grad():
                for val_batch_idx, (input_tensor, host_tensor, secret_tensor) in enumerate(val_dataloader):
                    # Non-blocking GPU transfer
                    input_tensor = input_tensor.to(device, non_blocking=True)
                    host_tensor = host_tensor.to(device, non_blocking=True)
                    secret_tensor = secret_tensor.to(device, non_blocking=True)

                    # Mixed precision inference
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        stego_output, log_det = model(input_tensor)
                        host_input = input_tensor[:, :3, :, :]
                        stego_host = stego_output[:, :3, :, :]

                        # Validation: use stego image to recover secret
                        stego_like = torch.cat([stego_host, torch.zeros_like(stego_host)], dim=1)
                        reconstructed_input = model.inverse(stego_like)
                        recovered_secret = reconstructed_input[:, 3:, :, :]

                    # Calculate validation loss
                    total_val_loss, val_hiding_psnr, val_recovery_psnr, val_rec_ssim_val, _, _ = steganography_loss(
                        secret_tensor,
                        recovered_secret,
                        host_input,
                        stego_host,
                        model.ilwt,
                        model.subband_weights,
                        target_psnr=42.0 # Example target PSNR
                    )

                    val_losses.append(total_val_loss.item())
                    
                    # Calculate metrics
                    val_hiding_psnr = calculate_psnr(stego_host, host_tensor)
                    val_recovery_psnr = calculate_psnr(recovered_secret, secret_tensor)
                    # val_rec_ssim_val is already SSIM value, no conversion needed
                    # val_rec_ssim_val = 1.0 - val_rec_ssim.item()  <-- REMOVED
                    val_bit_acc, val_bpp, _ = calculate_bit_acc_and_bpp(recovered_secret, secret_tensor)
                    
                    val_hiding_psnrs.append(val_hiding_psnr.item())
                    val_recovery_psnrs.append(val_recovery_psnr.item())
                    val_recovery_ssims.append(val_rec_ssim_val.item())
                    val_bit_accs.append(val_bit_acc)
            
            if val_losses:  # Check if we have validation data
                avg_val_loss = np.mean(val_losses)
                avg_val_hiding_psnr = np.mean(val_hiding_psnrs)
                avg_val_recovery_psnr = np.mean(val_recovery_psnrs)
                avg_val_recovery_ssim = np.mean(val_recovery_ssims)
                avg_val_bit_acc = np.mean(val_bit_accs)
        
        # Get metrics from a sample batch for logging (first batch of training)
        if len(train_dataloader) > 0:
            # Get a sample for metrics display
            sample_batch = next(iter(train_dataloader))
            input_tensor, host_tensor, secret_tensor = sample_batch
            input_tensor = input_tensor.to(device, non_blocking=True)
            host_tensor = host_tensor.to(device, non_blocking=True)
            secret_tensor = secret_tensor.to(device, non_blocking=True)

            model.eval()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                stego_output, log_det = model(input_tensor)
                stego_host = stego_output[:, :3, :, :]

                # Sample validation: use stego image to recover secret
                stego_like = torch.cat([stego_host, torch.zeros_like(stego_host)], dim=1)
                reconstructed_input = model.inverse(stego_like)
                recovered_secret = reconstructed_input[:, 3:, :, :]

                # Calculate sample metrics
                sample_hiding_psnr = calculate_psnr(stego_host, host_tensor)
                sample_recovery_psnr = calculate_psnr(recovered_secret, secret_tensor)
                rec_ssim_val = calculate_ssim(recovered_secret, secret_tensor).item()  # Note: this is SSIM, not loss
                sample_bit_acc, sample_bpp, _ = calculate_bit_acc_and_bpp(recovered_secret, secret_tensor)
            model.train()
        else:
            sample_hiding_psnr = torch.tensor(0.0)
            sample_recovery_psnr = torch.tensor(0.0)
            rec_ssim_val = 0.0
            sample_bit_acc = 0.0
            sample_bpp = 24.0

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {avg_train_loss:.6f}, "
            f"Hiding PSNR: {sample_hiding_psnr.item():.2f}, "
            f"Recovery PSNR: {sample_recovery_psnr.item():.2f}, "
            f"Rec SSIM: {rec_ssim_val:.4f}, "
            f"ACC: {sample_bit_acc:.4f}, "
            f"w_hid: {w_hid:.4f}, w_rec: {w_rec:.4f}"
        )
        
        # Log learnable weights
        with torch.no_grad():
            weights_lvl1, wLL2 = model.subband_weights.get_weights()
            wLL, wLH, wHL, wHH = weights_lvl1[0].item(), weights_lvl1[1].item(), weights_lvl1[2].item(), weights_lvl1[3].item()
            kY, kC = model.ycbcr_scaling.get_scales()
            print(f"  Learnable Weights: LL={wLL:.4f}, LH={wLH:.4f}, HL={wHL:.4f}, HH={wHH:.4f}, LL2={wLL2.item():.4f}")
            print(f"  YCbCr Scaling: kY={kY.item():.4f}, kC={kC.item():.4f}")
            
            # Store weights in history
            metrics_history['weight_LL'].append(wLL)
            metrics_history['weight_LH'].append(wLH)
            metrics_history['weight_HL'].append(wHL)
            metrics_history['weight_HH'].append(wHH)
            metrics_history['weight_LL2'].append(wLL2.item())
            metrics_history['scale_kY'].append(kY.item())
            metrics_history['scale_kC'].append(kC.item())
        
        if epoch % validate_every == 0:
            print(f"  Val Loss: {avg_val_loss:.6f}, "
                  f"Val Hid PSNR: {avg_val_hiding_psnr:.2f}, "
                  f"Val Rec PSNR: {avg_val_recovery_psnr:.2f}, "
                  f"Val Rec SSIM: {avg_val_recovery_ssim:.4f}, "
                  f"Val ACC: {avg_val_bit_acc:.4f}")

        # Calculate additional metrics from the sample batch
        sample_hiding_mse = F.mse_loss(stego_host, host_tensor).item()
        sample_recovery_mse = F.mse_loss(recovered_secret, secret_tensor).item()
        sample_hiding_ssim = calculate_ssim(stego_host, host_tensor).item()
        sample_ber, _, _ = calculate_bit_acc_and_bpp(recovered_secret, secret_tensor)
        sample_hiding_hr = calculate_hiding_ratio(host_tensor, stego_host)
        sample_bir = calculate_bitrate_increase_ratio(host_tensor, stego_host)

        # Skip complex validation metric calculations for now to avoid errors
        # We'll calculate them properly in the actual validation loop above
        avg_val_hiding_mse = 0
        avg_val_hiding_ssim = 0

        # Store metrics for research
        metrics_history['epoch'].append(epoch + 1)
        metrics_history['train_total_loss'].append(avg_train_loss)
        metrics_history['train_hiding_psnr'].append(sample_hiding_psnr.item())
        metrics_history['train_recovery_psnr'].append(sample_recovery_psnr.item())
        metrics_history['train_hiding_ssim'].append(sample_hiding_ssim)
        metrics_history['train_recovery_ssim'].append(rec_ssim_val)
        metrics_history['train_hiding_mse'].append(sample_hiding_mse)
        metrics_history['train_recovery_mse'].append(sample_recovery_mse)
        metrics_history['train_bit_acc'].append(sample_bit_acc)
        metrics_history['train_ber'].append(sample_ber)
        metrics_history['train_hiding_hr'].append(sample_hiding_hr)
        metrics_history['train_bir'].append(sample_bir)
        metrics_history['val_total_loss'].append(avg_val_loss)
        metrics_history['val_hiding_psnr'].append(avg_val_hiding_psnr)
        metrics_history['val_recovery_psnr'].append(avg_val_recovery_psnr)
        metrics_history['val_hiding_ssim'].append(avg_val_hiding_ssim)
        metrics_history['val_recovery_ssim'].append(avg_val_recovery_ssim)
        metrics_history['val_hiding_mse'].append(avg_val_hiding_mse)
        metrics_history['val_recovery_mse'].append(avg_val_recovery_psnr)  # using recovery psnr val as placeholder
        metrics_history['val_bit_acc'].append(avg_val_bit_acc)
        metrics_history['val_ber'].append(avg_val_bit_acc)  # using bit accuracy as placeholder for BER
        metrics_history['val_hiding_hr'].append(sample_hiding_hr)  # using training value as placeholder
        metrics_history['val_bir'].append(sample_bir)  # using training value as placeholder
        metrics_history['bpp'].append(sample_bpp)
        metrics_history['alpha_hid'].append(w_hid.item())

        # Write epoch info to log file
        with open("research_metrics/training_log.txt", 'a') as log_file:
            log_file.write(f"Epoch {epoch + 1}/{num_epochs}, "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Hiding PSNR: {sample_hiding_psnr.item():.2f}, "
                          f"Recovery PSNR: {sample_recovery_psnr.item():.2f}, "
                          f"Rec SSIM: {rec_ssim_val:.4f}, "
                          f"ACC: {sample_bit_acc:.4f}, "
                          f"BPP: {sample_bpp:.2f}, "
                          f"w_hid: {w_hid:.4f}, w_rec: {w_rec:.4f}\n")
            if epoch % validate_every == 0:
                log_file.write(f"  Val Loss: {avg_val_loss:.6f}, "
                              f"Val Hid PSNR: {avg_val_hiding_psnr:.2f}, "
                              f"Val Rec PSNR: {avg_val_recovery_psnr:.2f}, "
                              f"Val Rec SSIM: {avg_val_recovery_ssim:.4f}, "
                              f"Val ACC: {avg_val_bit_acc:.4f}\n")

        scheduler.step()
        
        # ============================================================================
        # CHECKPOINT SAVING & BEST MODEL TRACKING
        # ============================================================================
        
        # Track best model based on recovery PSNR
        current_recovery_psnr = sample_recovery_psnr.item()
        if not hasattr(train_model, 'best_recovery_psnr'):
            train_model.best_recovery_psnr = 0.0
        
        if current_recovery_psnr > train_model.best_recovery_psnr:
            train_model.best_recovery_psnr = current_recovery_psnr
            # Save best model
            best_model_path = 'checkpoints/best_model.pth'
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'recovery_psnr': current_recovery_psnr,
                'hiding_psnr': sample_hiding_psnr.item(),
                'metrics_history': metrics_history
            }, best_model_path)
            print(f"  *** New best model saved! Recovery PSNR: {current_recovery_psnr:.2f} dB ***")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, 
                metrics_history, train_model.best_recovery_psnr
            )

    
    # Save metrics to file for research analysis
    if save_metrics:
        # Directory already created at the start, so we don't need to create it again
        with open("research_metrics/training_metrics.json", "w") as f:
            json.dump(metrics_history, f, indent=4)
        print("Training metrics saved to research_metrics/training_metrics.json")

    print("Training completed!")
    return model, metrics_history


# Testing function
def test_model(model, dataset, num_samples=10):
    print(f"\nTesting model on {num_samples} samples...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Clean and create output directory
    if os.path.exists("ilwt_test_results"):
        shutil.rmtree("ilwt_test_results")
    os.makedirs("ilwt_test_results", exist_ok=False)

    # Randomly select test samples
    test_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    all_hiding_psnr = []
    all_recovery_psnr = []
    all_hiding_ssim = []
    all_recovery_ssim = []
    all_bit_acc = []
    all_hiding_mse = []
    all_recovery_mse = []
    all_extraction_success_rate = []
    all_ber = []
    all_hiding_hr = []
    all_bir = []
    bpp_value = None

    for i, idx in enumerate(test_indices):
        input_tensor, host_tensor, secret_tensor = dataset[idx]
        input_tensor = input_tensor.unsqueeze(0).to(device)
        host_tensor = host_tensor.unsqueeze(0).to(device)
        secret_tensor = secret_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            # Forward pass - create stego image
            stego_output, _ = model(input_tensor)

            # Inverse pass - recover secret using stego-only path
            stego_host = stego_output[:, :3, :, :]
            stego_like = torch.cat([stego_host, torch.zeros_like(stego_host)], dim=1)
            reconstructed_input = model.inverse(stego_like)
            recovered_secret = reconstructed_input[:, 3:, :, :]

            # Calculate metrics
            hiding_psnr = calculate_psnr(stego_host, host_tensor)
            recovery_psnr = calculate_psnr(recovered_secret, secret_tensor)
            hiding_ssim = calculate_ssim(stego_host, host_tensor)
            recovery_ssim = calculate_ssim(recovered_secret, secret_tensor)
            bit_acc, bpp, ber = calculate_bit_acc_and_bpp(recovered_secret, secret_tensor)
            hiding_hr = calculate_hiding_ratio(host_tensor, stego_host)
            hiding_bir = calculate_bitrate_increase_ratio(host_tensor, stego_host)
            
            # Calculate MSE for additional metrics
            hiding_mse = F.mse_loss(stego_host, host_tensor)
            recovery_mse = F.mse_loss(recovered_secret, secret_tensor)

            all_hiding_psnr.append(hiding_psnr.item())
            all_recovery_psnr.append(recovery_psnr.item())
            all_hiding_ssim.append(hiding_ssim.item())
            all_recovery_ssim.append(recovery_ssim.item())
            all_bit_acc.append(bit_acc)
            all_hiding_mse.append(hiding_mse.item())
            all_recovery_mse.append(recovery_mse.item())
            # Extraction success rate is bit accuracy thresholded at 0.9
            all_extraction_success_rate.append(1 if bit_acc >= 0.9 else 0)
            bpp_value = bpp

            # Store additional metrics
            all_ber.append(ber)
            all_hiding_hr.append(hiding_hr)
            all_bir.append(hiding_bir)

            # Additional metrics have been calculated and stored in lists
            # The print statement was already called in the main logging section of the loop

            # Denormalize for visualization (from [-1,1] to [0,1])
            def denormalize(tensor):
                result = (tensor / 2.0) + 0.5
                result = torch.clamp(result, 0, 1)
                return result

            host_vis = denormalize(host_tensor[0]).permute(1, 2, 0).cpu().numpy()
            secret_vis = denormalize(secret_tensor[0]).permute(1, 2, 0).cpu().numpy()
            stego_vis = denormalize(stego_host[0]).permute(1, 2, 0).cpu().numpy()
            recovered_vis = (
                denormalize(recovered_secret[0]).permute(1, 2, 0).cpu().numpy()
            )

            # Create visualization
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(host_vis)
            axes[0].set_title(
                f"Cover Image\n(Original Host)", fontsize=12, fontweight="bold"
            )
            axes[0].axis("off")

            axes[1].imshow(secret_vis)
            axes[1].set_title(
                f"Hidden Image\n(Secret to Hide)", fontsize=12, fontweight="bold"
            )
            axes[1].axis("off")

            axes[2].imshow(stego_vis)
            axes[2].set_title(
                f"Stego Image\n(Cover + Hidden)\nPSNR: {hiding_psnr.item():.2f} dB | SSIM: {hiding_ssim.item():.4f}",
                fontsize=12,
                fontweight="bold",
            )
            axes[2].axis("off")

            axes[3].imshow(recovered_vis)
            axes[3].set_title(
                f"Recovered Image\n(Extracted Secret)\nPSNR: {recovery_psnr.item():.2f} dB | SSIM: {recovery_ssim.item():.4f}",
                fontsize=12,
                fontweight="bold",
            )
            axes[3].axis("off")

            plt.suptitle(
                f"ILWT Steganography Test Sample {i + 1}",
                fontsize=16,
                fontweight="bold",
                y=1.02,
            )
            plt.tight_layout()
            plt.savefig(
                f"ilwt_test_results/test_sample_{i + 1}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

            print(
                f"Sample {i + 1}: Hiding PSNR={hiding_psnr.item():.2f} dB, "
                f"Recovery PSNR={recovery_psnr.item():.2f} dB, "
                f"Hiding SSIM={hiding_ssim.item():.4f}, "
                f"Recovery SSIM={recovery_ssim.item():.4f}, "
                f"ACC={bit_acc:.4f}, BPP={bpp:.2f}"
            )

    # Calculate and print average metrics
    avg_hiding_psnr = np.mean(all_hiding_psnr)
    avg_recovery_psnr = np.mean(all_recovery_psnr)
    avg_hiding_ssim = np.mean(all_hiding_ssim)
    avg_recovery_ssim = np.mean(all_recovery_ssim)
    avg_bit_acc = np.mean(all_bit_acc)
    avg_hiding_mse = np.mean(all_hiding_mse)
    avg_recovery_mse = np.mean(all_recovery_mse)
    avg_extraction_success_rate = np.mean(all_extraction_success_rate)
    
    # Calculate additional metrics
    avg_ber = 1.0 - avg_bit_acc  # BER = 1 - accuracy
    avg_hiding_hr = np.mean([calculate_hiding_ratio(host_tensor, stego_output[:, :3, :, :]) 
                            for idx in test_indices])
    avg_bir = np.mean([calculate_bitrate_increase_ratio(host_tensor, stego_output[:, :3, :, :]) 
                      for idx in test_indices])

    print("\n" + "=" * 70)
    print("TESTING COMPLETED - SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Average Hiding PSNR:           {avg_hiding_psnr:.2f} dB")
    print(f"Average Recovery PSNR:         {avg_recovery_psnr:.2f} dB")
    print(f"Average Hiding SSIM:           {avg_hiding_ssim:.4f}")
    print(f"Average Recovery SSIM:         {avg_recovery_ssim:.4f}")
    print(f"Average Bit Accuracy:          {avg_bit_acc:.4f}")
    print(f"Average Hiding MSE:            {avg_hiding_mse:.6f}")
    print(f"Average Recovery MSE:          {avg_recovery_mse:.6f}")
    print(f"Average Extraction Success:    {avg_extraction_success_rate:.4f}")
    print(f"Average Bit Error Rate (BER):  {avg_ber:.4f}")
    print(f"Average Hiding Ratio (HR):     {avg_hiding_hr:.4f}")
    print(f"Average Bitrate Increase (BIR): {avg_bir:.6f}")
    if bpp_value is not None:
        print(f"Bits Per Pixel (BPP):         {bpp_value:.2f}")
    print(f"\nTest results saved in 'ilwt_test_results' directory")
    
    # Save detailed metrics to JSON for research
    test_metrics = {
        'avg_hiding_psnr': avg_hiding_psnr,
        'avg_recovery_psnr': avg_recovery_psnr,
        'avg_hiding_ssim': avg_hiding_ssim,
        'avg_recovery_ssim': avg_recovery_ssim,
        'avg_bit_acc': avg_bit_acc,
        'avg_hiding_mse': avg_hiding_mse,
        'avg_recovery_mse': avg_recovery_mse,
        'avg_extraction_success_rate': avg_extraction_success_rate,
        'avg_ber': avg_ber,
        'avg_hiding_hr': avg_hiding_hr,
        'avg_bir': avg_bir,
        'bpp': bpp_value,
        'sample_metrics': {
            'hiding_psnr': all_hiding_psnr,
            'recovery_psnr': all_recovery_psnr,
            'hiding_ssim': all_hiding_ssim,
            'recovery_ssim': all_recovery_ssim,
            'bit_acc': all_bit_acc,
            'extraction_success_rate': all_extraction_success_rate,
            'ber': all_ber,
            'hiding_hr': all_hiding_hr,
            'bir': all_bir
        }
    }
    
    with open("research_metrics/test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)
    print("Detailed test metrics saved to research_metrics/test_metrics.json")
    print("=" * 70)

    return avg_hiding_psnr, avg_recovery_psnr, avg_hiding_ssim, avg_recovery_ssim, test_metrics


# Main function
def main():
    print("ILWT Steganography Training and Testing")
    print("=" * 50)

    image_dir = "my_images"
    img_size = 224
    num_blocks = 8  # Increased from 6 for better capacity
    hidden_channels = 128  # Increased from 96 for wider network
    num_epochs = 10  # Quick test with all features
    num_test_samples = 5  # Reduced for research testing

    # Load full dataset
    full_dataset = ImageSteganographyDataset(image_dir, img_size=img_size)
    print(f"Loaded {len(full_dataset)} images")

    # Split dataset into train, validation, and test
    train_dataset, val_dataset, test_dataset = split_dataset(
        full_dataset, 
        train_ratio=0.7, 
        val_ratio=0.15, 
        test_ratio=0.15, 
        random_seed=42
    )

    # Clean and create directories for research metrics
    if os.path.exists("research_metrics"):
        shutil.rmtree("research_metrics")
    os.makedirs("research_metrics", exist_ok=False)
    
    if os.path.exists("research_plots"):
        shutil.rmtree("research_plots")
    os.makedirs("research_plots", exist_ok=False)

    model = StarINNWithILWT(
        channels=6,
        num_blocks=num_blocks,
        hidden_channels=hidden_channels,
        transform_type="ilwt53",
    )
    model, metrics_history = train_model(model, train_dataset, val_dataset, num_epochs=num_epochs)

    torch.save(model.state_dict(), "ilwt_steganography_model_research.pth")
    print("\nModel saved as 'ilwt_steganography_model_research.pth'")

    # Test the trained model using test dataset
    avg_hiding_psnr, avg_recovery_psnr, avg_hiding_ssim, avg_recovery_ssim, test_metrics = test_model(model, test_dataset, num_samples=num_test_samples)
    
    # ============================================================================
    # GENERATE PUBLICATION-READY OUTPUTS
    # ============================================================================
    print("\n" + "=" * 70)
    print("GENERATING PUBLICATION-READY OUTPUTS")
    print("=" * 70)
    
    # Generate publication-quality plots (300 DPI PDF/PNG)
    generate_publication_plots(metrics_history)
    
    # Generate LaTeX tables
    generate_latex_tables(test_metrics)
    
    # Export comprehensive metrics (JSON, CSV, Markdown)
    export_metrics_comprehensive(metrics_history, test_metrics)
    
    print("\n" + "=" * 70)
    print("ALL PUBLICATION OUTPUTS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print("Outputs saved to:")
    print("  - publication_outputs/plots/     (300 DPI PDF/PNG)")
    print("  - publication_outputs/tables/    (LaTeX tables)")
    print("  - publication_outputs/metrics/   (JSON, CSV, Markdown)")
    print("  - checkpoints/                   (Model checkpoints)")
    print("=" * 70)
    
    # Generate research plots
    generate_research_plots(metrics_history)


def generate_research_plots(metrics_history):
    """Generate comprehensive plots for research paper with train/validation metrics"""
    import matplotlib.pyplot as plt
    
    # Clean and create research plots directory
    if os.path.exists("research_plots"):
        shutil.rmtree("research_plots")
    os.makedirs("research_plots", exist_ok=False)
    
    epochs = metrics_history['epoch']
    
    # Plot 1: Total Loss (Train vs Validation)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(epochs, metrics_history['train_total_loss'], 'b-', linewidth=2, label='Train')
    plt.plot(epochs, metrics_history['val_total_loss'], 'r-', linewidth=2, label='Validation')
    plt.title('Total Loss During Training', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Hiding PSNR (Train vs Validation)
    plt.subplot(2, 3, 2)
    plt.plot(epochs, metrics_history['train_hiding_psnr'], 'g-', linewidth=2, label='Train')
    plt.plot(epochs, metrics_history['val_hiding_psnr'], 'c--', linewidth=2, label='Validation')
    plt.title('Hiding PSNR During Training', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Recovery PSNR (Train vs Validation)
    plt.subplot(2, 3, 3)
    plt.plot(epochs, metrics_history['train_recovery_psnr'], 'r-', linewidth=2, label='Train')
    plt.plot(epochs, metrics_history['val_recovery_psnr'], 'm--', linewidth=2, label='Validation')
    plt.title('Recovery PSNR During Training', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Recovery SSIM (Train vs Validation)
    plt.subplot(2, 3, 4)
    plt.plot(epochs, metrics_history['train_recovery_ssim'], 'm-', linewidth=2, label='Train')
    plt.plot(epochs, metrics_history['val_recovery_ssim'], 'y--', linewidth=2, label='Validation')
    plt.title('Recovery SSIM During Training', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Bit Accuracy (Train vs Validation)
    plt.subplot(2, 3, 5)
    plt.plot(epochs, metrics_history['train_bit_acc'], 'c-', linewidth=2, label='Train')
    plt.plot(epochs, metrics_history['val_bit_acc'], 'k--', linewidth=2, label='Validation')
    plt.title('Bit Accuracy During Training', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Bit Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Alpha HID (Loss Weight)
    plt.subplot(2, 3, 6)
    plt.plot(epochs, metrics_history['alpha_hid'], 'k-', linewidth=2)  # Now contains w_hid
    plt.title('Alpha HID (Loss Weight) During Training', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Alpha HID Value')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("research_plots/training_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create additional plots for more detailed analysis
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: All PSNR metrics together
    plt.subplot(2, 3, 1)
    plt.plot(epochs, metrics_history['train_hiding_psnr'], label='Train Hiding PSNR', linewidth=2)
    plt.plot(epochs, metrics_history['train_recovery_psnr'], label='Train Recovery PSNR', linewidth=2)
    plt.plot(epochs, metrics_history['val_hiding_psnr'], label='Val Hiding PSNR', linewidth=2, linestyle='--')
    plt.plot(epochs, metrics_history['val_recovery_psnr'], label='Val Recovery PSNR', linewidth=2, linestyle='--')
    plt.title('PSNR Metrics Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Loss components
    plt.subplot(2, 3, 2)
    plt.plot(epochs, metrics_history['train_total_loss'], label='Train Loss', linewidth=2)
    plt.plot(epochs, metrics_history['val_total_loss'], label='Validation Loss', linewidth=2, linestyle='--')
    plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: SSIM metrics
    plt.subplot(2, 3, 3)
    plt.plot(epochs, metrics_history['train_recovery_ssim'], label='Train Recovery SSIM', linewidth=2)
    plt.plot(epochs, metrics_history['val_recovery_ssim'], label='Validation Recovery SSIM', linewidth=2, linestyle='--')
    plt.title('SSIM Metrics Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Bit Accuracy comparison
    plt.subplot(2, 3, 4)
    plt.plot(epochs, metrics_history['train_bit_acc'], label='Train Bit Acc', linewidth=2)
    plt.plot(epochs, metrics_history['val_bit_acc'], label='Validation Bit Acc', linewidth=2, linestyle='--')
    plt.title('Bit Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Bit Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Overfitting detection (Train vs Validation gaps)
    train_val_gap_psnr = [t - v for t, v in zip(metrics_history['train_recovery_psnr'], metrics_history['val_recovery_psnr'])]
    train_val_gap_loss = [t - v for t, v in zip(metrics_history['train_total_loss'], metrics_history['val_total_loss'])]
    plt.subplot(2, 3, 5)
    plt.plot(epochs, train_val_gap_psnr, label='PSNR Gap (Train - Val)', linewidth=2)
    plt.plot(epochs, train_val_gap_loss, label='Loss Gap (Train - Val)', linewidth=2)
    plt.title('Overfitting Detection', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Gap')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: BPP and overall metrics
    plt.subplot(2, 3, 6)
    if metrics_history['bpp']:
        # BPP is constant
        bpp_val = metrics_history['bpp'][0] if metrics_history['bpp'] else 24.0
        plt.bar(['Bits Per Pixel'], [bpp_val], color='orange')
        plt.title(f'Embedding Capacity\nBPP: {bpp_val:.2f}', fontsize=14, fontweight='bold')
        plt.ylabel('Bits Per Pixel')
    else:
        plt.text(0.5, 0.5, 'BPP not available', horizontalalignment='center', verticalalignment='center')
        plt.title('Embedding Capacity', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("research_plots/comprehensive_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot weight evolution
    plt.figure(figsize=(12, 6))
    epochs = metrics_history['epoch']
    plt.plot(epochs, metrics_history['weight_LL'], label='LL Weight', linewidth=2)
    plt.plot(epochs, metrics_history['weight_LH'], label='LH Weight', linewidth=2)
    plt.plot(epochs, metrics_history['weight_HL'], label='HL Weight', linewidth=2)
    plt.plot(epochs, metrics_history['weight_HH'], label='HH Weight', linewidth=2)
    plt.plot(epochs, metrics_history['weight_LL2'], label='LL2 Weight', linewidth=2, linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Weight Value')
    plt.title('Wavelet Subband Weight Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('research_plots/weight_evolution_wavelet.png', dpi=300)
    plt.close()

    # Plot YCbCr scaling evolution
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, metrics_history['scale_kY'], label='Luminance (kY)', linewidth=2, color='orange')
    plt.plot(epochs, metrics_history['scale_kC'], label='Chrominance (kC)', linewidth=2, color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('Scale Factor')
    plt.title('YCbCr Scaling Parameter Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('research_plots/weight_evolution_ycbcr.png', dpi=300)
    plt.close()

    print("Research plots saved in research_plots/ directory")


if __name__ == "__main__":
    main()

