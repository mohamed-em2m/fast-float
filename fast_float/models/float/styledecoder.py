import math

import torch
from torch import nn
from torch.nn import functional as F

# ============================================================================
# LEVEL 1: Low-level utility functions
# ============================================================================


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2**0.5):
    """Supports both single and batched inputs via broadcasting"""
    return F.leaky_relu(input + bias, negative_slope) * scale


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    """Already supports batched inputs naturally"""
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
    """Already supports batched inputs"""
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])


def make_kernel(k):
    """Kernel creation - no batching needed"""
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


# ============================================================================
# LEVEL 2: Basic modules (already batch-compatible)
# ============================================================================


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2**0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        """Already batch-compatible via broadcasting"""
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        """Already batch-compatible"""
        return input * torch.rsqrt(torch.mean(input**2, dim=1, keepdim=True) + 1e-8)


class MotionPixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        """Already batch-compatible"""
        return input * torch.rsqrt(torch.mean(input**2, dim=2, keepdim=True) + 1e-8)


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

    def forward(self, input):
        """Already batch-compatible"""
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
        """Already batch-compatible"""
        return upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)
        self.register_buffer("kernel", kernel)
        self.pad = pad

    def forward(self, input):
        """Already batch-compatible"""
        return upfirdn2d(input, self.kernel, pad=self.pad)


# ============================================================================
# LEVEL 3: Convolution layers (already batch-compatible)
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
        """Already batch-compatible"""
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
        """Pre-multiply static scale and lr_mul into weight/bias parameters for export/inference."""
        with torch.no_grad():
            if self.scale != 1.0:
                self.weight.copy_(self.weight * self.scale)
                self.scale = 1.0
            if self.bias is not None and self.lr_mul != 1.0:
                self.bias.copy_(self.bias * self.lr_mul)
                self.lr_mul = 1.0

    def forward(self, input):
        """Already batch-compatible"""
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
        """Already batch-compatible"""
        return F.leaky_relu(input, negative_slope=self.negative_slope)


# ============================================================================
# LEVEL 4: Advanced modules with batch handling
# ============================================================================


class ModulatedConv2d(nn.Module):
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

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        """Enhanced to handle both single and batch inputs"""
        batch, in_channel, height, width = input.shape

        # Modulation - already batch compatible
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)

        if self.upsample:
            input = input.contiguous().view(1, batch * in_channel, height, width)
            weight = weight.view(batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
            weight = (
                weight.transpose(1, 2)
                .contiguous()
                .reshape(
                    batch * in_channel,
                    self.out_channel,
                    self.kernel_size,
                    self.kernel_size,
                )
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)
        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.contiguous().view(1, batch * in_channel, height, width)
            weight = weight.contiguous()
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        else:
            input = input.contiguous().view(1, batch * in_channel, height, width)
            weight = weight.contiguous()
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        """Already batch-compatible"""
        if noise is None:
            return image
        else:
            return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        """Enhanced to handle variable batch sizes"""
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out


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
        """Already batch-compatible"""
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

    def forward(self, input, skip=None):
        """Already batch-compatible"""
        out = self.conv(input)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out


def _identity_grid(h: int, w: int, device, dtype) -> torch.Tensor:
    """Normalized ``[1, H, W, 2]`` sampling grid for ``grid_sample``.

    Built functionally (no module state), so ``torch.compile`` never sees a
    mutated attribute. The tensors are small (H*W*2) compared with the feature
    maps they warp, so rebuilding them per call costs well under a millisecond.
    """
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device, dtype=dtype),
        torch.linspace(-1, 1, w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)


class ToFlow(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=None):
        if blur_kernel is None:
            blur_kernel = [1, 3, 3, 1]
        super().__init__()
        if upsample:
            self.upsample = Upsample(blur_kernel)
        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, feat, skip=None):
        """Batch-compatible forward with a stateless coordinate grid.

        The grid depends only on (h, w, dtype). It used to be built lazily and
        stored with ``register_buffer`` *inside* forward, which made the module
        mutate while tracing: TorchDynamo's ``hasattr(module, '_cached_grid')``
        guard failed right after the first call and forced a full recompile of
        the decoder (the single biggest part of boot time). Build it before the
        trace instead — the tensors are tiny and created once per shape.
        """
        batch_size = input.size(0)
        h, w = input.size(2), input.size(3)
        grid = _identity_grid(h, w, input.device, input.dtype).expand(batch_size, -1, -1, -1)

        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        # Flow computation
        sampler = torch.tanh(out[:, 0:2, :, :])
        mask = torch.sigmoid(out[:, 2:3, :, :])
        flow = sampler.permute(0, 2, 3, 1) + grid  # B x h x w x 2

        # --- Ensure batch sizes match for grid_sample ---
        if feat.shape[0] != batch_size:
            if feat.shape[0] == 1:
                # Expand instead of repeat (more efficient, no memory copy)
                feat = feat.expand(batch_size, -1, -1, -1)
            else:
                raise ValueError(f"Batch mismatch: feat batch={feat.shape[0]}, flow batch={batch_size}")

        # Mask batch check (optimized)
        if mask.shape[0] != batch_size:
            raise ValueError(f"Mask batch mismatch: mask batch={mask.shape[0]}, feat batch={batch_size}")
        # Optional: ensure mask batch matches too
        if mask.shape[0] != feat.shape[0]:
            raise ValueError(f"Mask batch mismatch: mask batch={mask.shape[0]}, feat batch={feat.shape[0]}")

        # --- Warp features ---
        feat_warp = F.grid_sample(feat, flow, align_corners=False) * mask

        return feat_warp, feat_warp + input * (1.0 - mask), out, flow


# ============================================================================
# LEVEL 6: Direction module with batch handling
# ============================================================================


class Direction(nn.Module):
    def __init__(self, motion_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(512, motion_dim))
        # Q depends only on the (frozen) weight, but forward() recomputed
        # torch.linalg.qr on EVERY call (once per decode micro-batch).
        # Cache it. Invalidation is a cheap fingerprint (version + data_ptr +
        # fp32 sum over ~10K elems): .data.copy_ bypasses the version counter,
        # so version alone is not reliable.
        self._Q_cache = None
        self._Q_fp = None

    def _fingerprint(self, w):
        # Deliberately no `.item()`: reading a scalar back forces a device sync
        # on every forward, and it is what Dynamo reports as a graph break here
        # (`Tensor.item()`), splitting the compiled decoder in two. This is
        # still complete for every loader in the repo: in-place overlays
        # (`param.copy_`, `load_state_dict`) bump `_version`, and replacing the
        # parameter changes `data_ptr`. Only a raw `.data.copy_` would slip
        # through, and nothing on the inference path does that.
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
        """Enhanced to handle both single and batch inputs"""
        Q = self._orthonormal_basis()

        if input is None:
            return Q
        else:
            # Handle both shapes: [motion_dim] and [batch, motion_dim]
            if input.dim() == 1:
                input = input.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False

            # Batch matrix multiplication
            input_diag = torch.diag_embed(input)  # [batch, motion_dim, motion_dim]
            out = torch.matmul(input_diag, Q.T)  # [batch, motion_dim, 512]
            out = torch.sum(out, dim=1)  # [batch, 512]

            if squeeze_output:
                out = out.squeeze(0)

            return out


# ============================================================================
# LEVEL 7: Top-level Synthesis module with full batch support
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

    def forward(self, wa, alpha, feats):
        """
        Enhanced forward pass supporting both single and batch inputs

        Args:
            wa: [batch, 512] or [512] - style latent codes
            alpha: list of motion coefficients, each [batch, motion_dim] or [motion_dim]
            feats: list of feature maps for warping

        Returns:
            img: Generated image(s)
            flow: Optical flow field(s)
        """
        # Ensure wa is batched
        if wa.dim() == 1:
            wa = wa.unsqueeze(0)
            single_input = True
        else:
            single_input = False

        wa.shape[0]

        # Apply motion directions
        if alpha is not None:
            if len(alpha) > 1:
                # Multi-direction case: target, source, start
                directions_target = self.direction(alpha[0])
                directions_source = self.direction(alpha[1])
                directions_start = self.direction(alpha[2])
                latent = wa + (directions_target - directions_start) + directions_source
            else:
                # Single direction case
                directions = self.direction(alpha[0])
                latent = wa + directions
        else:
            latent = wa

        # Expand latent to all layers
        inject_index = self.n_latent
        latent = latent.unsqueeze(1).repeat(1, inject_index, 1)

        # Initial layers
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0])

        # Progressive generation with flow warping
        i = 1
        skip = None
        skip_flow = None
        flow = None

        for conv1, conv2, to_rgb, to_flow, feat in zip(
            self.convs[::2], self.convs[1::2], self.to_rgbs, self.to_flows, feats, strict=False
        ):
            out = conv1(out, latent[:, i])
            out = conv2(out, latent[:, i + 1])

            # Apply flow warping at specific resolutions
            if out.size(2) == 8:
                out_warp, out, skip_flow, _ = to_flow(out, latent[:, i + 2], feat)
                skip = to_rgb(out_warp)
            elif out.size(2) == 64:
                out_warp, out, skip_flow, flow = to_flow(out, latent[:, i + 2], feat, skip_flow)
                skip = to_rgb(out_warp, skip)
            else:
                out_warp, out, skip_flow, _ = to_flow(out, latent[:, i + 2], feat, skip_flow)
                skip = to_rgb(out_warp, skip)

            i += 2

        img = skip

        # If input was single, squeeze output
        if single_input:
            img = img.squeeze(0)
            if flow is not None:
                flow = flow.squeeze(0)

        return img, flow

    def synthesis(self, wa, alpha, feats):
        """
        Alias for forward() to maintain backward compatibility
        Supports both single and batch processing
        """
        return self.forward(wa, alpha, feats)
