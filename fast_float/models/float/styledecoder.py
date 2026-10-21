"""styledecoder.py - StyleGAN2-based talking portrait decoder for FLOAT.

Contains high-performance inference implementations for StyleDecoder modules,
eliminating grouped convolutions (groups=batch), using separable depthwise FIR
filtering, commuted last-level feature warping, and precomputed orthonormal bases.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

# ============================================================================
# LEVEL 1: Low-level utility functions
# ============================================================================


def _hp(t: torch.Tensor) -> torch.Tensor:
    """Detached copy at >= fp32 precision (keeps float64 as float64)."""
    t = t.detach().clone()
    return t.float() if t.dtype in (torch.float16, torch.bfloat16) else t


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2**0.5):
    """Supports both single and batched inputs via broadcasting."""
    return F.leaky_relu(input + bias, negative_slope) * scale


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    """Fallback native implementation of upfirdn2d."""
    _, minor, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape
    out = input.view(-1, minor, in_h, 1, in_w, 1)
    out = F.pad(out, [0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0])
    out = out.view(-1, minor, in_h * up_y, in_w * up_x)
    out = F.pad(out, [max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[
        :,
        :,
        max(-pad_y0, 0) : out.shape[2] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[3] - max(-pad_x1, 0),
    ]
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    return out[:, :, ::down_y, ::down_x]


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """Batched upfirdn2d wrapper."""
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])


def make_kernel(k):
    """Kernel creation helper."""
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


def _identity_grid(h: int, w: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Normalized [1, H, W, 2] sampling grid for grid_sample."""
    if isinstance(device, torch.dtype):
        dtype, device = device, None
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device, dtype=dtype),
        torch.linspace(-1, 1, w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)


# ============================================================================
# LEVEL 2: Basic modules (FIR & normalization)
# ============================================================================


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2**0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input**2, dim=1, keepdim=True) + 1e-8)


class MotionPixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input**2, dim=2, keepdim=True) + 1e-8)


class FIRBlur(nn.Module):
    """Depthwise conv FIR blur (separable 1D convs when possible)."""

    def __init__(self, kernel2d: torch.Tensor, pad):
        super().__init__()
        p0, p1 = pad
        assert p0 >= 0, "negative padding is not supported"
        assert p1 >= 0, "negative padding is not supported"
        self.pad = (int(p0), int(p1))
        flipped = torch.flip(_hp(kernel2d), [0, 1])
        u, s, vh = torch.linalg.svd(flipped)
        self.separable = bool(s.numel() < 2 or s[1:].max() <= 1e-6 * s[0])
        if self.separable:
            root = s[0].sqrt()
            self.register_buffer("k_col", (u[:, 0] * root).view(1, 1, -1, 1).contiguous(), persistent=False)
            self.register_buffer("k_row", (vh[0] * root).view(1, 1, 1, -1).contiguous(), persistent=False)
        else:
            self.register_buffer("k2d", flipped.view(1, 1, *flipped.shape).contiguous(), persistent=False)

    def forward(self, x):
        c = x.shape[1]
        p0, p1 = self.pad
        x = F.pad(x, (p0, p1, p0, p1))
        if self.separable:
            k_row = self.k_row.to(dtype=x.dtype, device=x.device)
            k_col = self.k_col.to(dtype=x.dtype, device=x.device)
            x = F.conv2d(x, k_row.expand(c, 1, 1, -1), groups=c)
            return F.conv2d(x, k_col.expand(c, 1, -1, 1), groups=c)
        k2d = self.k2d.to(dtype=x.dtype, device=x.device)
        return F.conv2d(x, k2d.expand(c, 1, -1, -1), groups=c)


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)
        self.register_buffer("kernel", kernel)
        self.pad = pad
        p0, p1 = pad
        assert p0 >= 0, "negative padding is not supported"
        assert p1 >= 0, "negative padding is not supported"
        flipped = torch.flip(kernel.detach().float(), [0, 1])
        u, s, vh = torch.linalg.svd(flipped)
        self.separable = bool(s.numel() < 2 or s[1:].max() <= 1e-6 * s[0])
        if self.separable:
            root = s[0].sqrt()
            self.register_buffer("k_col", (u[:, 0] * root).view(1, 1, -1, 1).contiguous(), persistent=False)
            self.register_buffer("k_row", (vh[0] * root).view(1, 1, 1, -1).contiguous(), persistent=False)
        else:
            self.register_buffer("k2d", flipped.view(1, 1, *flipped.shape).contiguous(), persistent=False)

    def forward(self, input):
        c = input.shape[1]
        p0, p1 = self.pad
        x = F.pad(input, (p0, p1, p0, p1))
        if self.separable:
            k_row = self.k_row.to(dtype=x.dtype, device=x.device)
            k_col = self.k_col.to(dtype=x.dtype, device=x.device)
            x = F.conv2d(x, k_row.expand(c, 1, 1, -1), groups=c)
            return F.conv2d(x, k_col.expand(c, 1, -1, 1), groups=c)
        k2d = self.k2d.to(dtype=x.dtype, device=x.device)
        return F.conv2d(x, k2d.expand(c, 1, -1, -1), groups=c)


class FIRUp2(nn.Module):
    """Depthwise conv_transpose2d upsampler."""

    def __init__(self, up_module):
        super().__init__()
        assert up_module.factor == 2
        kernel = _hp(up_module.kernel)  # already scaled by factor**2
        kk = kernel.shape[0]
        p0, _ = up_module.pad
        self.crop = kk - 1 - p0
        crop_end = kk - 2 - self.crop
        assert self.crop == crop_end, "asymmetric crop: unsupported Upsample padding"
        assert self.crop >= 0, "asymmetric crop: unsupported Upsample padding"
        self.register_buffer("k", kernel.view(1, 1, kk, kk).contiguous(), persistent=False)

    def forward(self, x):
        c = x.shape[1]
        k = self.k.to(dtype=x.dtype, device=x.device)
        return F.conv_transpose2d(x, k.expand(c, 1, -1, -1), stride=2, padding=self.crop, groups=c)


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        kernel = make_kernel(kernel) * (factor**2)
        self.register_buffer("kernel", kernel)
        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2
        self.pad = (pad0, pad1)
        kk = kernel.shape[0]
        self.crop = kk - 1 - pad0
        self.register_buffer("k_up", kernel.view(1, 1, kk, kk).contiguous(), persistent=False)

    def forward(self, input):
        if self.factor == 2 and self.pad[0] == (self.kernel.shape[0] - self.factor + 1) // 2 + self.factor - 1:
            c = input.shape[1]
            k = self.k_up.to(dtype=input.dtype, device=input.device)
            return F.conv_transpose2d(input, k.expand(c, 1, -1, -1), stride=2, padding=self.crop, groups=c)
        return upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)
        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2
        pad1 = p // 2
        self.pad = (pad0, pad1)

    def forward(self, input):
        return upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)


# ============================================================================
# LEVEL 3: Convolution layers
# ============================================================================


class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)
        self.stride = stride
        self.padding = padding
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def fold_weights(self):
        """Pre-multiply static scale into weight parameter for export/inference."""
        with torch.no_grad():
            if self.scale != 1.0:
                self.weight.copy_(self.weight * self.scale)
                self.scale = 1.0

    def forward(self, input):
        w = self.weight * self.scale if self.scale != 1.0 else self.weight
        return F.conv2d(
            input,
            w,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def fold_weights(self):
        """Pre-multiply static scale and lr_mul into weight/bias parameters."""
        with torch.no_grad():
            if self.scale != 1.0:
                self.weight.copy_(self.weight * self.scale)
                self.scale = 1.0
            if self.bias is not None and self.lr_mul != 1.0:
                self.bias.copy_(self.bias * self.lr_mul)
                self.lr_mul = 1.0

    def forward(self, input):
        w = self.weight * self.scale if self.scale != 1.0 else self.weight
        b = self.bias
        if b is not None and self.lr_mul != 1.0:
            b = b * self.lr_mul
        if self.activation:
            out = F.linear(input, w)
            out = fused_leaky_relu(out, b)
        else:
            out = F.linear(input, w, bias=b)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        return F.leaky_relu(input, negative_slope=self.negative_slope)


# ============================================================================
# LEVEL 4: Advanced modules with batch handling
# ============================================================================


class ModulatedConv2d(nn.Module):
    """Dense modulated convolution without groups=batch.

    Linearly scales the input channels by the style vector, applies a static dense conv,
    and scales the output channels by demodulation d = rsqrt(style^2 @ w2 + eps).
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=None,
    ):
        if blur_kernel is None:
            blur_kernel = [1, 3, 3, 1]
        super().__init__()
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size**2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate
        self._w2_cache = None
        self._w2_fp = None

    def _get_w2(self, w):
        fp = (w._version, w.data_ptr(), w.device, w.dtype)
        if self._w2_cache is None or self._w2_fp != fp:
            self._w2_cache = w.pow(2).sum((2, 3)).t().contiguous()
            self._w2_fp = fp
        return self._w2_cache

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, _, _, _ = input.shape

        style = self.modulation(style)
        input = input * style.view(batch, -1, 1, 1)

        w = (self.scale * self.weight[0]).to(dtype=input.dtype, device=input.device)

        if self.upsample:
            wt = w.transpose(0, 1).contiguous()
            out = F.conv_transpose2d(input, wt, stride=2, padding=0)
            out = self.blur(out)
        elif self.downsample:
            out = self.blur(input)
            out = F.conv2d(out, w, padding=0, stride=2)
        else:
            out = F.conv2d(input, w, padding=self.padding)

        if self.demodulate:
            w2 = self._get_w2(w)
            acc = torch.promote_types(style.dtype, torch.float32)
            demod = torch.rsqrt(style.to(acc).square() @ w2.to(acc) + self.eps)
            out = out * demod.to(out.dtype).view(batch, self.out_channel, 1, 1)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            return image
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        return self.input.expand(batch, -1, -1, -1)


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=None,
        demodulate=True,
    ):
        if blur_kernel is None:
            blur_kernel = [1, 3, 3, 1]
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )
        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)
        return out


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=None,
        bias=True,
        activate=True,
    ):
        if blur_kernel is None:
            blur_kernel = [1, 3, 3, 1]
        layers = []
        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


# ============================================================================
# LEVEL 5: Output layers with batch handling
# ============================================================================


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=None):
        if blur_kernel is None:
            blur_kernel = [1, 3, 3, 1]
        super().__init__()
        if upsample:
            self.upsample = Upsample(blur_kernel)
        self.conv = ConvLayer(in_channel, 3, 1)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def project(self, x):
        """Linear 1x1 EqualConv2d part (commutes with warp * mask)."""
        eq = self.conv[0]
        w = (eq.weight * eq.scale).to(dtype=x.dtype, device=x.device)
        return F.conv2d(x, w)

    def finish(self, z, skip=None):
        """FusedLeakyReLU activation + bias + upsampled skip."""
        act = self.conv[1]
        act_bias = act.bias.to(dtype=z.dtype, device=z.device)
        bias = self.bias.to(dtype=z.dtype, device=z.device)
        out = F.leaky_relu(z + act_bias, act.negative_slope) * act.scale + bias
        if skip is not None and hasattr(self, "upsample"):
            out = out + self.upsample(skip)
        return out

    def forward(self, input, skip=None):
        out = self.conv(input) + self.bias
        if skip is not None and hasattr(self, "upsample"):
            skip = self.upsample(skip)
            out = out + skip
        return out


class ToFlow(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=None):
        if blur_kernel is None:
            blur_kernel = [1, 3, 3, 1]
        super().__init__()
        if upsample:
            self.upsample = Upsample(blur_kernel)
        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def flow_and_mask(self, input, style, skip=None):
        batch_size, _, h, w = input.shape
        out = self.conv(input, style) + self.bias
        if skip is not None and hasattr(self, "upsample"):
            out = out + self.upsample(skip)
        sampler = torch.tanh(out[:, 0:2, :, :])
        mask = torch.sigmoid(out[:, 2:3, :, :])
        grid = _identity_grid(h, w, input.device, input.dtype).expand(batch_size, -1, -1, -1)
        flow = sampler.permute(0, 2, 3, 1) + grid
        return out, flow, mask

    def forward(self, input, style, feat, skip=None):
        batch_size = input.size(0)
        out, flow, mask = self.flow_and_mask(input, style, skip)
        if feat.shape[0] != batch_size:
            if feat.shape[0] == 1:
                feat = feat.expand(batch_size, -1, -1, -1)
            else:
                raise ValueError(f"Batch mismatch: feat batch={feat.shape[0]}, flow batch={batch_size}")
        feat_warp = F.grid_sample(feat, flow, align_corners=False) * mask
        return feat_warp, feat_warp + input * (1.0 - mask), out, flow


# ============================================================================
# LEVEL 6: Direction module with batch handling
# ============================================================================


class Direction(nn.Module):
    def __init__(self, motion_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(512, motion_dim))
        self._Q_cache = None
        self._Q_fp = None

    def _fingerprint(self, w):
        return (w._version, w.data_ptr(), w.device, w.dtype)

    def _orthonormal_basis(self):
        """Cached Q factor of weight. Recomputed only if weight changed."""
        w = self.weight
        fp = self._fingerprint(w)
        if self._Q_cache is None or self._Q_fp != fp:
            Q, _ = torch.linalg.qr(w + 1e-8)
            self._Q_cache = Q
            self._Q_fp = fp
        return self._Q_cache

    def forward(self, input):
        """Matrix multiplication input @ Q.T supporting arbitrary batch shapes."""
        Q = self._orthonormal_basis()
        if input is None:
            return Q
        q = Q.to(dtype=input.dtype, device=input.device)
        if input.dim() == 1:
            return (input.unsqueeze(0) @ q.t()).squeeze(0)
        return input @ q.t()


# ============================================================================
# LEVEL 7: Top-level Synthesis module
# ============================================================================


class Synthesis(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        motion_dim,
        blur_kernel=None,
        channel_multiplier=1,
    ):
        if blur_kernel is None:
            blur_kernel = [1, 3, 3, 1]
        super().__init__()
        self.size = size
        self.style_dim = style_dim
        self.motion_dim = motion_dim

        self.direction = Direction(motion_dim)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.to_flows = nn.ModuleList()

        in_channel = self.channels[4]
        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2**i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )
            self.convs.append(StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel))
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            self.to_flows.append(ToFlow(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2
        self.n_levels = len(self.to_flows)

    def precompute(self, feats):
        """Per-identity work: 3-channel RGB projection of the last warped feature map."""
        if feats is None or len(feats) == 0:
            return None
        n = min(len(feats), self.n_levels)
        return self.to_rgbs[n - 1].project(feats[n - 1])

    def forward(self, wa, alpha=None, feats=None, z_last=None):
        if feats is None and isinstance(alpha, (list, tuple)):
            feats = alpha
            alpha = None

        if wa.dim() == 1:
            wa = wa.unsqueeze(0)
            single_input = True
        else:
            single_input = False

        if alpha is not None and len(alpha) > 0:
            if len(alpha) >= 3:
                latent = wa + (self.direction(alpha[0]) - self.direction(alpha[2])) + self.direction(alpha[1])
            elif len(alpha) == 2:
                latent = wa + (self.direction(alpha[0]) + self.direction(alpha[1]))
            else:
                latent = wa + self.direction(alpha[0])
        else:
            latent = wa

        b = latent.shape[0]
        out = self.input(latent)
        out = self.conv1(out, latent)

        n = min(len(feats), self.n_levels) if feats is not None else 0
        skip = skip_flow = flow = None

        for k in range(n):
            conv1 = self.convs[2 * k]
            conv2 = self.convs[2 * k + 1]
            to_flow = self.to_flows[k]
            to_rgb = self.to_rgbs[k]
            feat = feats[k]

            out = conv1(out, latent)
            out = conv2(out, latent)

            if k == n - 1:
                # Last level: commute 1x1 conv to warp 3 channels instead of C channels
                skip_flow, flw, mask = to_flow.flow_and_mask(out, latent, skip_flow)
                z = z_last if z_last is not None else to_rgb.project(feat)
                z = z.to(device=out.device, dtype=out.dtype)
                if z.shape[0] != b:
                    if z.shape[0] == 1:
                        z = z.expand(b, -1, -1, -1)
                    else:
                        raise ValueError(f"Batch mismatch: z_last batch={z.shape[0]}, flow batch={b}")
                skip = to_rgb.finish(F.grid_sample(z, flw, align_corners=False) * mask, skip)
            else:
                out_warp, out, skip_flow, flw = to_flow(out, latent, feat, skip_flow)
                skip = to_rgb(out_warp, skip)

            if flw.shape[1] == 64:  # flow at 64x64
                flow = flw

        img = skip
        if single_input and img is not None:
            img = img.squeeze(0)
            if flow is not None:
                flow = flow.squeeze(0)

        return img, flow

    def synthesis(self, wa, alpha=None, feats=None, z_last=None):
        return self.forward(wa, alpha=alpha, feats=feats, z_last=z_last)

    def to_infer(self, fold_blur: bool = False, dtype=None, channels_last: bool = False):
        """Convert this Synthesis module into an optimized SynthesisInfer module."""
        return SynthesisInfer.from_synthesis(self, fold_blur=fold_blur, dtype=dtype, channels_last=channels_last)


# ============================================================================
# LEVEL 8: Static-Buffer Inference Modules (SynthesisInfer)
# ============================================================================


class ModConvInfer(nn.Module):
    def __init__(self, mc, fold_blur: bool = False):
        super().__init__()
        assert not mc.downsample, "downsample ModulatedConv2d is not supported"
        w = _hp(mc.scale * mc.weight[0])  # [O, I, k, k]
        self.demodulate = bool(mc.demodulate)
        self.upsample = bool(mc.upsample)
        self.padding = int(mc.padding)
        self.fold_pad = 0
        self.blur = None
        if self.demodulate:
            self.register_buffer("w2", w.pow(2).sum((2, 3)).t().contiguous(), persistent=False)  # [I, O]
        if self.upsample:
            wt = w.transpose(0, 1).contiguous()  # [I, O, k, k] (conv_transpose layout)
            kk = mc.blur.kernel.shape[0]
            k = w.shape[-1]
            p0, _ = mc.blur.pad
            crop = kk - 1 - p0
            comp = k + kk - 1
            foldable = fold_blur and crop >= 0 and 2 * crop == comp - 2
            if foldable:
                i, o = wt.shape[:2]
                kern = _hp(mc.blur.kernel)
                full = F.conv2d(
                    wt.reshape(i * o, 1, k, k),
                    torch.flip(kern, [0, 1]).view(1, 1, kk, kk),
                    padding=kk - 1,
                )
                self.register_buffer(
                    "w_conv", full.reshape(i, o, k + kk - 1, k + kk - 1).contiguous(), persistent=False
                )
                self.fold_pad = crop
            else:
                self.register_buffer("w_conv", wt, persistent=False)
                self.blur = FIRBlur(mc.blur.kernel, mc.blur.pad)
        else:
            self.register_buffer("w_conv", w, persistent=False)

    def forward(self, x, s):
        b = x.shape[0]
        x = x * s.view(b, -1, 1, 1)
        w_conv = self.w_conv.to(dtype=x.dtype, device=x.device)
        if self.upsample:
            out = F.conv_transpose2d(x, w_conv, stride=2, padding=self.fold_pad)
            if self.blur is not None:
                out = self.blur(out)
        else:
            out = F.conv2d(x, w_conv, padding=self.padding)
        if self.demodulate:
            acc = torch.promote_types(s.dtype, torch.float32)
            w2 = self.w2.to(dtype=acc, device=s.device)
            d = torch.rsqrt(s.to(acc).square() @ w2 + 1e-8)
            out = out * d.to(out.dtype).view(b, -1, 1, 1)
        return out


class StyledConvInfer(nn.Module):
    def __init__(self, sc, fold_blur=False):
        super().__init__()
        self.conv = ModConvInfer(sc.conv, fold_blur=fold_blur)
        self.register_buffer("bias", sc.activate.bias.detach().clone(), persistent=False)
        self.neg = float(sc.activate.negative_slope)
        self.scale = float(sc.activate.scale)

    def forward(self, x, s):
        out = self.conv(x, s)
        bias = self.bias.to(dtype=out.dtype, device=out.device)
        return F.leaky_relu(out + bias, self.neg) * self.scale


class ToRGBInfer(nn.Module):
    def __init__(self, tr):
        super().__init__()
        eq, act = tr.conv[0], tr.conv[1]
        assert eq.bias is None, "bias in 1x1 conv is not supported"
        assert eq.weight.shape[-1] == 1, "non-1x1 conv is not supported"
        assert eq.stride == 1, "strided conv is not supported"
        self.register_buffer("w", (eq.weight * eq.scale).detach().clone(), persistent=False)
        self.register_buffer("act_bias", act.bias.detach().clone(), persistent=False)
        self.register_buffer("bias", tr.bias.detach().clone(), persistent=False)
        self.neg = float(act.negative_slope)
        self.act_scale = float(act.scale)
        self.up = FIRUp2(tr.upsample) if hasattr(tr, "upsample") else None

    def project(self, x):
        w = self.w.to(dtype=x.dtype, device=x.device)
        return F.conv2d(x, w)

    def finish(self, z, skip=None):
        act_bias = self.act_bias.to(dtype=z.dtype, device=z.device)
        bias = self.bias.to(dtype=z.dtype, device=z.device)
        out = F.leaky_relu(z + act_bias, self.neg) * self.act_scale + bias
        if skip is not None:
            out = out + (self.up(skip) if self.up is not None else skip)
        return out

    def forward(self, x, skip=None):
        return self.finish(self.project(x), skip)


class ToFlowInfer(nn.Module):
    def __init__(self, tf, res: int):
        super().__init__()
        self.conv = ModConvInfer(tf.conv)
        self.register_buffer("bias", tf.bias.detach().clone(), persistent=False)
        self.up = FIRUp2(tf.upsample) if hasattr(tf, "upsample") else None
        self.register_buffer("grid", _identity_grid(res, res, dtype=_hp(tf.bias).dtype), persistent=False)

    def flow_and_mask(self, x, s, skip=None):
        b, _, h, w = x.shape
        bias = self.bias.to(dtype=x.dtype, device=x.device)
        out = self.conv(x, s) + bias
        if skip is not None:
            out = out + (self.up(skip) if self.up is not None else skip)
        sampler = torch.tanh(out[:, 0:2])
        mask = torch.sigmoid(out[:, 2:3])
        if self.grid.shape[1] == h and self.grid.shape[2] == w:
            grid = self.grid.to(dtype=x.dtype, device=x.device)
        else:
            grid = _identity_grid(h, w, dtype=x.dtype).to(x.device)
        flow = sampler.permute(0, 2, 3, 1) + grid.expand(b, -1, -1, -1)
        return out, flow, mask

    def forward(self, x, s, feat, skip=None):
        b = x.shape[0]
        out, flow, mask = self.flow_and_mask(x, s, skip)
        if feat.shape[0] != b:
            if feat.shape[0] != 1:
                raise ValueError(f"Batch mismatch: feat batch={feat.shape[0]}, flow batch={b}")
            feat = feat.expand(b, -1, -1, -1)
        feat_warp = F.grid_sample(feat, flow, align_corners=False) * mask
        return feat_warp, feat_warp + x * (1.0 - mask), out, flow


class DirectionInfer(nn.Module):
    def __init__(self, direction):
        super().__init__()
        self.weight = getattr(direction, "weight", None)
        q, _ = torch.linalg.qr(_hp(direction.weight) + 1e-8)
        self.register_buffer("Q", q.contiguous(), persistent=False)

    def forward(self, input):
        if input is None:
            return self.Q
        q = self.Q.to(dtype=input.dtype, device=input.device)
        if input.dim() == 1:
            return (input.unsqueeze(0) @ q.t()).squeeze(0)
        return input @ q.t()


class SynthesisInfer(nn.Module):
    """Drop-in inference decoder with pre-fused single-GEMM styles and frozen weights."""

    def __init__(self, syn, fold_blur: bool = False):
        super().__init__()
        self.size = syn.size
        self.style_dim = syn.style_dim
        self.motion_dim = syn.motion_dim
        self.channels = getattr(syn, "channels", None)
        self.n_latent = getattr(syn, "n_latent", None)
        self.n_levels = len(syn.to_flows)
        self._ff_channels_last = False

        self.direction = DirectionInfer(syn.direction)
        self.register_buffer("const", syn.input.input.detach().clone(), persistent=False)

        self.conv1 = StyledConvInfer(syn.conv1, fold_blur=fold_blur)
        self.convs_a = nn.ModuleList()
        self.convs_b = nn.ModuleList()
        self.to_flows = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        mods = [syn.conv1.conv.modulation]
        for k in range(self.n_levels):
            self.convs_a.append(StyledConvInfer(syn.convs[2 * k], fold_blur=fold_blur))
            self.convs_b.append(StyledConvInfer(syn.convs[2 * k + 1], fold_blur=fold_blur))
            self.to_flows.append(ToFlowInfer(syn.to_flows[k], res=2 ** (k + 3)))
            self.to_rgbs.append(ToRGBInfer(syn.to_rgbs[k]))
            mods += [
                syn.convs[2 * k].conv.modulation,
                syn.convs[2 * k + 1].conv.modulation,
                syn.to_flows[k].conv.modulation,
            ]

        assert all(m.activation is None and m.bias is not None for m in mods)
        w = torch.cat([(m.weight * m.scale).detach() for m in mods], dim=0)
        b = torch.cat([(m.bias * m.lr_mul).detach() for m in mods], dim=0)
        self.register_buffer("style_w", w.contiguous(), persistent=False)
        self.register_buffer("style_b", b.contiguous(), persistent=False)
        self.style_sizes = [int(m.weight.shape[0]) for m in mods]

    @classmethod
    @torch.no_grad()
    def from_synthesis(cls, syn, fold_blur: bool = False, dtype=None, channels_last: bool = False):
        if isinstance(syn, cls):
            if dtype is not None:
                syn.to(dtype)
            if channels_last:
                syn.channels_last()
            return syn
        params = list(syn.parameters())
        if params:
            dev = params[0].device
        else:
            bufs = list(syn.buffers())
            dev = bufs[0].device if bufs else torch.device("cpu")
        model = cls(syn, fold_blur=fold_blur).to(dev).eval()
        model.requires_grad_(False)
        if dtype is not None:
            model.to(dtype)
        if channels_last:
            model.channels_last()
        return model

    def channels_last(self):
        for m in self.modules():
            for name in ("w_conv", "w", "const"):
                t = getattr(m, name, None)
                if isinstance(t, torch.Tensor) and t.dim() == 4:
                    setattr(m, name, t.contiguous(memory_format=torch.channels_last))
        self._ff_channels_last = True
        return self

    def precompute(self, feats):
        """Per-identity work: RGB projection of the last warped feature map."""
        if feats is None or len(feats) == 0:
            return None
        n = min(len(feats), self.n_levels)
        return self.to_rgbs[n - 1].project(feats[n - 1])

    def forward(self, wa, alpha=None, feats=None, z_last=None):
        if feats is None and isinstance(alpha, (list, tuple)):
            feats = alpha
            alpha = None

        single = wa.dim() == 1
        if single:
            wa = wa.unsqueeze(0)

        if alpha is not None and len(alpha) > 0:
            if len(alpha) >= 3:
                latent = wa + (self.direction(alpha[0]) - self.direction(alpha[2])) + self.direction(alpha[1])
            elif len(alpha) == 2:
                latent = wa + (self.direction(alpha[0]) + self.direction(alpha[1]))
            else:
                latent = wa + self.direction(alpha[0])
        else:
            latent = wa

        b = latent.shape[0]
        style_w = self.style_w.to(dtype=latent.dtype, device=latent.device)
        style_b = self.style_b.to(dtype=latent.dtype, device=latent.device)
        styles = F.linear(latent, style_w, style_b).split(self.style_sizes, dim=1)

        const = self.const.to(dtype=latent.dtype, device=latent.device)
        out = const.expand(b, -1, -1, -1)
        out = self.conv1(out, styles[0])

        n = min(len(feats), self.n_levels) if feats is not None else 0
        skip = skip_flow = flow = None
        si = 1
        for k in range(n):
            out = self.convs_a[k](out, styles[si])
            out = self.convs_b[k](out, styles[si + 1])
            s_flow = styles[si + 2]
            si += 3
            to_flow, to_rgb, feat = self.to_flows[k], self.to_rgbs[k], feats[k]

            if k == n - 1:
                skip_flow, flw, mask = to_flow.flow_and_mask(out, s_flow, skip_flow)
                z = z_last if z_last is not None else to_rgb.project(feat)
                z = z.to(device=out.device, dtype=out.dtype)
                if z.shape[0] != b:
                    if z.shape[0] == 1:
                        z = z.expand(b, -1, -1, -1)
                    else:
                        raise ValueError(f"Batch mismatch: z_last batch={z.shape[0]}, flow batch={b}")
                skip = to_rgb.finish(F.grid_sample(z, flw, align_corners=False) * mask, skip)
            else:
                out_warp, out, skip_flow, flw = to_flow(out, s_flow, feat, skip_flow)
                skip = to_rgb(out_warp, skip)

            if flw.shape[1] == 64:
                flow = flw

        img = skip
        if single and img is not None:
            img = img.squeeze(0)
            if flow is not None:
                flow = flow.squeeze(0)
        return img, flow

    def synthesis(self, wa, alpha=None, feats=None, z_last=None):
        return self.forward(wa, alpha=alpha, feats=feats, z_last=z_last)


__all__ = [
    "Blur",
    "ConstantInput",
    "ConvLayer",
    "Direction",
    "DirectionInfer",
    "Downsample",
    "EqualConv2d",
    "EqualLinear",
    "FIRBlur",
    "FIRUp2",
    "FusedLeakyReLU",
    "ModConvInfer",
    "ModulatedConv2d",
    "MotionPixelNorm",
    "NoiseInjection",
    "PixelNorm",
    "ScaledLeakyReLU",
    "StyledConv",
    "StyledConvInfer",
    "Synthesis",
    "SynthesisInfer",
    "ToFlow",
    "ToFlowInfer",
    "ToRGB",
    "ToRGBInfer",
    "Upsample",
    "fused_leaky_relu",
    "make_kernel",
    "upfirdn2d",
    "upfirdn2d_native",
]
