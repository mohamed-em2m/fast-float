import os
import math
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None
from fast_float.models import BaseModel

try:
    from timm.layers import use_fused_attn
    from timm.models.vision_transformer import Mlp
except Exception:

    def use_fused_attn():
        return hasattr(F, "scaled_dot_product_attention")

    class Mlp(nn.Module):
        def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
        ):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.drop1 = nn.Dropout(drop)
            self.norm = nn.Identity()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop2 = nn.Dropout(drop)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop1(x)
            x = self.fc2(x)
            x = self.drop2(x)
            return x


logger = logging.getLogger(__name__)


def enc_dec_mask(T, S, frame_width=1, expansion=2):
    mask = torch.ones(T, S)
    for i in range(T):
        mask[
            i, max(0, (i - expansion) * frame_width) : (i + expansion + 1) * frame_width
        ] = 0
    return mask == 1


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """
    Sinusoidal position encoding table.
    Args:
            n_position (int): the length of the input sequence
            d_hid (int): the dimension of the hidden state
    """

    def cal_angle(position, hid_idx):
        return position / (10000 ** (2 * (hid_idx // 2) / d_hid))

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = torch.Tensor(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.0
    return sinusoid_table


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:

        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=~mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float32)
        elif t.ndim == 0:
            t = t.unsqueeze(0)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SequenceEmbed(nn.Module):
    def __init__(
        self,
        dim_w,
        dim_h,
        norm_layer=None,
        bias=True,
    ):
        super().__init__()

        self.proj = nn.Linear(dim_w, dim_h, bias=bias)
        self.norm = norm_layer(dim_h) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class FMTBlock(nn.Module):
    """
    A FMT block inspried by DiT Block
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def framewise_modulate(self, x, shift, scale) -> torch.Tensor:
        return x * (1 + scale) + shift

    def forward(self, x, c, mask=None) -> torch.Tensor:
        assert mask is not None
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(
            self.framewise_modulate(self.norm1(x), shift_msa, scale_msa), mask=mask
        )
        x = x + gate_mlp * self.mlp(
            self.framewise_modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class Decoder(nn.Module):
    """
    The final decoder of FlowMatchingTransformer.
    """

    def __init__(self, hidden_size, dim_w):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, dim_w, bias=True)

    def framewise_modulate(self, x, shift, scale) -> torch.Tensor:
        return x * (1 + scale) + shift

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.framewise_modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class FlowMatchingTransformer(BaseModel):
    """
    Flow Matching Transformer (FMT)
    """

    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

        self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(opt.num_prev_frames)
        self.num_total_frames = self.num_prev_frames + self.num_frames_for_clip

        self.hidden_size = opt.dim_h
        self.mlp_ratio = opt.mlp_ratio
        self.fmt_depth = opt.fmt_depth
        self.num_heads = opt.num_heads

        self.x_embedder = SequenceEmbed(opt.dim_w, self.hidden_size)

        # video time position encoding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_total_frames, self.hidden_size), requires_grad=False
        )

        # flow trajectory time encoding
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.c_embedder = nn.Linear(opt.dim_w + opt.dim_a + opt.dim_e, self.hidden_size)

        # define FMT blocks
        self.blocks = nn.ModuleList(
            [
                FMTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio)
                for _ in range(self.fmt_depth)
            ]
        )
        self.decoder = Decoder(self.hidden_size, self.opt.dim_w)
        self.initialize_weights()

        # define alignment mask
        _mask_dev = torch.device(
            f"cuda:{opt.rank}" if torch.cuda.is_available() else "cpu"
        )
        alignment_mask = enc_dec_mask(
            self.num_total_frames,
            self.num_total_frames,
            1,
            expansion=opt.attention_window,
        ).to(_mask_dev)
        self.register_buffer("alignment_mask", alignment_mask)

        # Prefer the ONNX single-step backend (fast). The ODE loop, CFG
        # combine and chunk autoregression stay in Python; only the inner
        # vector-field eval is routed through ORT (see _forward_step).
        # NOTE: parity is validated by the converter/selector scripts, never
        # automatically here -- the config is the contract.
        self._onnx_backend = None
        self._compiled = None
        try:
            from fast_float.onnx.backends import resolve_mode

            self._mode = resolve_mode("fmt")
        except Exception:
            self._mode = "eager"
        onnx_path = getattr(opt, "fmt_onnx_path", "./onnx/fmt_step_v1.onnx")
        if self._mode == "onnx" and onnx_path and os.path.exists(onnx_path):
            try:
                from fast_float.onnx.runtime import FMTStepBackend

                self._onnx_backend = FMTStepBackend(onnx_path)
                logger.info(
                    f"FlowMatchingTransformer: ONNX backend constructed {onnx_path} "
                    f"(providers={self._onnx_backend.providers})"
                )
            except Exception as e:
                logger.warning(
                    f"FlowMatchingTransformer: ONNX backend failed ({e}); "
                    "falling back to torch weights"
                )
                self._onnx_backend = None

    def _probe_onnx_backend(self, backend, tol: float = 1e-3) -> None:
        """Self-test helper for scripts/manual use only (never called here).

        Runs on CUDA when available (deploy device; also unifies the
        alignment-mask buffer, which is created on ``opt.rank``, with the
        parameters for the reference eval).
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        g = torch.Generator(device="cpu").manual_seed(0)
        kw = dict(dtype=torch.float32)
        t = torch.tensor([0.5], **kw)
        x = torch.randn(1, 50, 512, generator=g, **kw)
        wa = torch.randn(1, 50, 512, generator=g, **kw)
        wr = torch.randn(1, 1, 512, generator=g, **kw)
        we = torch.randn(1, 1, 7, generator=g, **kw)
        prev_x = torch.randn(1, 10, 512, generator=g, **kw)
        prev_wa = torch.randn(1, 10, 512, generator=g, **kw)
        t, x, wa, wr, we, prev_x, prev_wa = (
            v.to(device) for v in (t, x, wa, wr, we, prev_x, prev_wa)
        )
        with torch.no_grad():
            ref = self.forward(t, x, wa, wr, we, prev_x, prev_wa, train=False)
            got = backend.run_step(t, x, wa, wr, we, prev_x, prev_wa)
        err = (ref.float().cpu() - got.float().cpu()).abs().max().item()
        print(f"[fmt-onnx] init probe max_abs={err:.3e} (tol={tol:g})")
        if not bool(torch.isfinite(torch.tensor(err))) or err > tol:
            raise RuntimeError(f"ONNX probe mismatch (max_abs={err:.3e})")

    @property
    def uses_onnx(self) -> bool:
        return self._onnx_backend is not None

    def _forward_step(self, t, x, wa, wr, we, prev_x, prev_wa):
        """Inner vector-field eval: ORT when engaged, else torch.

        No automatic parity probe here by design -- validation belongs to the
        converter/selector scripts; the config is the contract.
        """
        backend = self._onnx_backend
        if backend is not None and all(
            torch.is_tensor(v) for v in (t, x, wa, wr, we, prev_x, prev_wa)
        ):
            try:
                return backend.run_step(t, x, wa, wr, we, prev_x, prev_wa)
            except Exception as e:
                logger.warning(
                    f"FlowMatchingTransformer: ONNX step failed ({e}); "
                    "disabling backend for this run"
                )
                self._onnx_backend = None
        if self._mode == "compile":
            if self._compiled is None:
                try:
                    from fast_float.onnx.backends import maybe_compile

                    self._compiled = maybe_compile(self, "fmt")
                except Exception as e:
                    logger.warning(
                        f"FlowMatchingTransformer: torch.compile failed ({e}); "
                        "using eager"
                    )
                    self._compiled = self
            if self._compiled is not self:
                return self._compiled(t, x, wa, wr, we, prev_x, prev_wa, train=False)
        return self.forward(t, x, wa, wr, we, prev_x, prev_wa, train=False)

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_sinusoid_encoding_table(self.num_total_frames, self.hidden_size)
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in FMT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.decoder.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.linear.weight, 0)
        nn.init.constant_(self.decoder.linear.bias, 0)

    def sequence_embedder(self, sequence, dropout_prob, train=False) -> torch.Tensor:
        if train:
            batch_id_for_drop = torch.where(
                torch.rand(sequence.shape[0], device=sequence.device) < dropout_prob
            )
            sequence[batch_id_for_drop] = 0
        return sequence

    def forward(
        self, t, x, wa, wr, we, prev_x=None, prev_wa=None, train=True, **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of ConditionalFlowMatchingTransformer supporting batch size B >= 1.

        t: (B,) or scalar tensor of diffusion timesteps [0, 1]
        x: (B, L, 512) : tensor of sequence of motion latent

        wa:  (B, L, 512)  / tensor sequence of wa latent
        wr:  (B, 512) or (B, 1, 512) / tensor of reference motion latent
        we:  (B, 1, 7) or (B, 7) / tensor of emotion latent

        prev_x:  (B, L', 512) / previous x for auto-regressive generation
        prev_wa: (B, L', 512) / previous audio for auto-regressive generation
        """
        B = x.shape[0]

        # time encoding
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=x.device, dtype=torch.float32)
        elif t.ndim == 0:
            t = t.unsqueeze(0)
        if t.device != x.device:
            t = t.to(x.device)
        t = self.t_embedder(t).unsqueeze(1)  # (N_t, 1, D)
        if t.shape[0] != B and t.shape[0] == 1:
            t = t.expand(B, -1, -1)

        # condition encoding & batch dimension alignment
        if wa.ndim == 2:
            wa = wa.unsqueeze(0)
        if wa.shape[0] != B:
            wa = wa.expand(B, -1, -1)
        wa = self.sequence_embedder(
            wa, dropout_prob=self.opt.audio_dropout_prob, train=train
        )

        if wr.ndim == 1:
            wr = wr.unsqueeze(0).unsqueeze(1)
        elif wr.ndim == 2:
            wr = wr.unsqueeze(1)
        if wr.shape[0] != B:
            wr = wr.expand(B, -1, -1)
        wr = self.sequence_embedder(
            wr, dropout_prob=self.opt.ref_dropout_prob, train=train
        )

        if we.ndim == 1:
            we = we.unsqueeze(0).unsqueeze(1)
        elif we.ndim == 2:
            we = we.unsqueeze(1)
        if we.shape[0] != B:
            we = we.expand(B, -1, -1)
        we = self.sequence_embedder(
            we, dropout_prob=self.opt.emotion_dropout_prob, train=train
        )

        # previous condition encoding
        if prev_x is not None:
            if prev_x.ndim == 2:
                prev_x = prev_x.unsqueeze(0)
            if prev_x.shape[0] != B:
                prev_x = prev_x.expand(B, -1, -1)
            prev_x = self.sequence_embedder(prev_x, dropout_prob=0.5, train=train)

        if prev_wa is not None:
            if prev_wa.ndim == 2:
                prev_wa = prev_wa.unsqueeze(0)
            if prev_wa.shape[0] != B:
                prev_wa = prev_wa.expand(B, -1, -1)
            prev_wa = self.sequence_embedder(prev_wa, dropout_prob=0.5, train=train)

        if prev_x is not None:
            x = torch.cat([prev_x, x], dim=1)
        if prev_wa is not None:
            wa = torch.cat([prev_wa, wa], dim=1)

        x = self.x_embedder(x)
        x = (
            x + self.pos_embed
        )  # (B, L + L', D), where T = opt.wav2vec_sec * opt.fps, D = dim_w

        # expand (view, no copy) instead of repeat: the cat below copies once.
        wr = wr.expand(-1, wa.shape[1], -1)
        we = we.expand(-1, wa.shape[1], -1)

        c = torch.cat([wr, wa, we], dim=-1)
        c = self.c_embedder(c)
        c = t + c

        # forwarding FMT Blocks
        for block in self.blocks:
            x = block(x, c, self.alignment_mask)  # (B, T, D)
        return self.decoder(x, c)

    @torch.no_grad()
    def forward_with_cfv(
        self,
        t,
        x,
        wa,
        wr,
        we,
        prev_x=None,
        prev_wa=None,
        a_cfg_scale=1.0,
        r_cfg_scale=1.0,
        e_cfg_scale=1.0,
        # ---- pre-built constant tensors (optional) ----
        # Pass these from the caller to avoid re-allocating on every ODE step:
        #   _ref_cat     : torch.cat([wr, wr, wr], 0)          - constant all inference
        #   _emotion_cat : torch.cat([null_we, we, null_we], 0) - constant all inference
        #   _audio_cat   : torch.cat([null_wa, wa, wa], 0)      - constant within a chunk
        #   _prev_x_cat  : torch.cat([prev_x]*3, 0)             - constant within a chunk
        #   _prev_wa_cat : torch.cat([prev_wa]*3, 0)            - constant within a chunk
        _ref_cat=None,
        _emotion_cat=None,
        _audio_cat=None,
        _prev_x_cat=None,
        _prev_wa_cat=None,
        **kwargs,
    ) -> torch.Tensor:
        if a_cfg_scale != 1.0 or r_cfg_scale != 1.0 or e_cfg_scale != 1.0:
            B = x.shape[0]

            # ---- audio cat (changes every chunk, pre-built per chunk) ----
            if _audio_cat is not None:
                audio_cat = _audio_cat
            else:
                if wa.ndim == 2:
                    wa = wa.unsqueeze(0)
                if wa.shape[0] != B:
                    wa = wa.expand(B, -1, -1)
                null_wa = torch.zeros_like(wa)
                audio_cat = torch.cat([null_wa, wa, wa], dim=0)

            # ---- ref cat (constant for entire inference) ----
            if _ref_cat is not None:
                ref_cat = _ref_cat
            else:
                if wr.ndim == 1:
                    wr = wr.unsqueeze(0).unsqueeze(1)
                elif wr.ndim == 2:
                    wr = wr.unsqueeze(1)
                if wr.shape[0] != B:
                    wr = wr.expand(B, -1, -1)
                ref_cat = torch.cat([wr, wr, wr], dim=0)

            # ---- emotion cat (constant for entire inference) ----
            if _emotion_cat is not None:
                emotion_cat = _emotion_cat
            else:
                if we.ndim == 1:
                    we = we.unsqueeze(0).unsqueeze(1)
                elif we.ndim == 2:
                    we = we.unsqueeze(1)
                if we.shape[0] != B:
                    we = we.expand(B, -1, -1)
                null_we = torch.zeros_like(we)
                emotion_cat = torch.cat([null_we, we, null_we], dim=0)

            # ---- x (changes every ODE step, must always be built here) ----
            x_cat = torch.cat([x, x, x], dim=0)

            # ---- prev tensors (constant within a chunk) ----
            if _prev_x_cat is not None:
                prev_x_cat = _prev_x_cat
            elif prev_x is not None:
                if prev_x.ndim == 2:
                    prev_x = prev_x.unsqueeze(0)
                if prev_x.shape[0] != B:
                    prev_x = prev_x.expand(B, -1, -1)
                prev_x_cat = torch.cat([prev_x, prev_x, prev_x], dim=0)
            else:
                prev_x_cat = None

            if _prev_wa_cat is not None:
                prev_wa_cat = _prev_wa_cat
            elif prev_wa is not None:
                if prev_wa.ndim == 2:
                    prev_wa = prev_wa.unsqueeze(0)
                if prev_wa.shape[0] != B:
                    prev_wa = prev_wa.expand(B, -1, -1)
                prev_wa_cat = torch.cat([prev_wa, prev_wa, prev_wa], dim=0)
            else:
                prev_wa_cat = None

            # Replicate t along batch if t was already explicitly sized to B
            if torch.is_tensor(t) and t.ndim > 0 and t.shape[0] == B and B > 1:
                t_cat = torch.cat([t, t, t], dim=0)
            else:
                t_cat = t

            model_output = self._forward_step(
                t_cat,
                x_cat,
                audio_cat,
                ref_cat,
                emotion_cat,
                prev_x_cat,
                prev_wa_cat,
            )
            uncond, all_cond, audio_uncond_emotion = torch.chunk(
                model_output, chunks=3, dim=0
            )

            # Classifier-free vector field (cfv) incremental manner
            return (
                uncond
                + a_cfg_scale * (audio_uncond_emotion - uncond)
                + e_cfg_scale * (all_cond - audio_uncond_emotion)
            )
        else:
            return self._forward_step(t, x, wa, wr, we, prev_x, prev_wa)
