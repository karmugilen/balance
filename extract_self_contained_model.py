import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np


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
        kY=0.01,
        kC=0.04,
    ):
        super(StarINNWithILWT, self).__init__()
        # Initialize learnable parameters
        self.subband_weights = LearnableSubbandWeights()
        self.ycbcr_scaling = LearnableYCbCrScaling(init_kY=kY, init_kC=kC)

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


def load_image_for_tensor(path, size):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open(path).convert('RGB')
    tensor = transform(img)
    return tensor


def denormalize_to_pil(tensor):
    tensor = (tensor / 2.0) + 0.5
    tensor = torch.clamp(tensor, 0.0, 1.0)
    img = transforms.ToPILImage()(tensor.cpu())
    return img


def main():
    parser = argparse.ArgumentParser(description="Extract a secret image from a stego image using a trained ILWT model")
    parser.add_argument("--model", required=True, help="Path to trained .pth file")
    parser.add_argument("--stego", required=True, help="Path to stego image (PNG/JPG)")
    parser.add_argument("--output", required=True, help="Output recovered secret image path (PNG)")
    parser.add_argument("--size", type=int, default=256, help="Square size for preprocessing (default: 256)")
    parser.add_argument("--num_blocks", type=int, default=8, help="Number of StarINN blocks (must match training)")
    parser.add_argument("--hidden_channels", type=int, default=128, help="Hidden channels in coupling nets (match training)")
    parser.add_argument("--transform_type", type=str, default="ilwt53", choices=["ilwt53", "haar_conv"], help="Transform backend used in training")
    parser.add_argument("--kY", type=float, default=0.01, help="Y channel scaling factor (default: 0.01)")
    parser.add_argument("--kC", type=float, default=0.04, help="Cb/Cr channel scaling factor (default: 0.04)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    # Load model
    model = StarINNWithILWT(
        channels=6, 
        num_blocks=args.num_blocks, 
        hidden_channels=args.hidden_channels, 
        transform_type=args.transform_type,
        kY=args.kY,
        kC=args.kC
    )
    # Load model (handle both checkpoint and state_dict)
    state = torch.load(args.model, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        print(f"Loading from checkpoint (Epoch {state.get('epoch', 'unknown')})")
        model.load_state_dict(state['model_state_dict'])
    else:
        print("Loading from state dict")
        model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    # Prepare stego-like input from PNG: [stego, zeros]
    stego_tensor = load_image_for_tensor(args.stego, args.size).unsqueeze(0).to(device)
    stego_like = torch.cat([stego_tensor, torch.zeros_like(stego_tensor)], dim=1)

    with torch.no_grad():
        reconstructed_input = model.inverse(stego_like)
        recovered_secret = reconstructed_input[:, 3:, :, :][0]

    recovered_img = denormalize_to_pil(recovered_secret)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    recovered_img.save(args.output)
    print(f"Saved recovered secret to {args.output}")


if __name__ == "__main__":
    main()