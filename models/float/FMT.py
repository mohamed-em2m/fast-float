import os, math, torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from models import BaseModel
from timm.layers import use_fused_attn
from timm.models.vision_transformer import Mlp


def enc_dec_mask(T, S, frame_width=1, expansion=2):
    """
    Creates a diagonal mask for local attention.
    """
    mask = torch.ones(T, S)
    for i in range(T):
        start = max(0, (i - expansion) * frame_width)
        end = (i + expansion + 1) * frame_width
        mask[i, start:end] = 0
    return mask == 1


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """
    Sinusoidal position encoding table.
    """
    def cal_angle(position, hid_idx):
        return position / (10000 ** (2 * (hid_idx // 2) / d_hid))
    
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
    
    sinusoid_table = torch.Tensor([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])
    
    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.
    
    return sinusoid_table


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if self.fused_attn:
            # Mask needs to be boolean for scaled_dot_product_attention
            if mask is not None:
                # Expand mask: (N, N) -> (B, num_heads, N, N)
                attn_mask = mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)
                attn_mask = ~attn_mask  # Invert: True = masked out
            else:
                attn_mask = None
                
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            
            if mask is not None:
                # Expand mask for batch and heads: (N, N) -> (B, num_heads, N, N)
                mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)
                attn = attn.masked_fill(mask_expanded, float('-inf'))
            
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        Handles scalar, 1D, or batched inputs robustly.
        """
        # Handle scalar or 0D tensor
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        # Ensure t is 1D
        if t.dim() > 1:
            original_shape = t.shape
            t = t.reshape(-1)
            reshape_needed = True
        else:
            reshape_needed = False
            
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        if reshape_needed:
            embedding = embedding.reshape(*original_shape, -1)
            
        return embedding
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SequenceEmbed(nn.Module):
    def __init__(self, dim_w, dim_h, norm_layer=None, bias=True):
        super().__init__()
        self.proj = nn.Linear(dim_w, dim_h, bias=bias)
        self.norm = norm_layer(dim_h) if norm_layer else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class FMTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
    
    def framewise_modulate(self, x, shift, scale) -> torch.Tensor:
        return x * (1 + scale) + shift
    
    def forward(self, x, c, mask=None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        
        x_norm = self.norm1(x)
        x_mod = self.framewise_modulate(x_norm, shift_msa, scale_msa)
        x = x + gate_msa * self.attn(x_mod, mask=mask)
        
        x_norm2 = self.norm2(x)
        x_mod2 = self.framewise_modulate(x_norm2, shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_mod2)
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_size, dim_w):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
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
    Flow Matching Transformer (FMT) with Full Batching Support
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
        
        # Position Encoding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_total_frames, self.hidden_size), 
            requires_grad=False
        )
        
        # Time & Condition Embedders
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.c_embedder = nn.Linear(opt.dim_w + opt.dim_a + opt.dim_e, self.hidden_size)
        
        # Blocks
        self.blocks = nn.ModuleList([
            FMTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio) 
            for _ in range(self.fmt_depth)
        ])
        
        self.decoder = Decoder(self.hidden_size, self.opt.dim_w)
        
        self.initialize_weights()
        
        # Pre-compute alignment mask (device-agnostic initialization)
        alignment_mask = enc_dec_mask(
            self.num_total_frames, 
            self.num_total_frames, 
            1, 
            expansion=opt.attention_window
        )
        self.register_buffer('alignment_mask', alignment_mask)
    
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
        
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.decoder.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.linear.weight, 0)
        nn.init.constant_(self.decoder.linear.bias, 0)
    
    def sequence_embedder(self, sequence, dropout_prob, train=False) -> torch.Tensor:
        """
        Apply sequence-level dropout (for classifier-free guidance).
        Supports arbitrary batch sizes and sequence dimensions.
        """
        if train and dropout_prob > 0:
            device = sequence.device
            
            # Determine mask shape based on input dimensions
            if sequence.dim() == 3:  # (B, L, D)
                mask_shape = (sequence.shape[0], 1, 1)
            elif sequence.dim() == 2:  # (B, D)
                mask_shape = (sequence.shape[0], 1)
            else:
                return sequence  # No dropout for unexpected dimensions
            
            keep_prob = 1 - dropout_prob
            mask = (torch.rand(mask_shape, device=device) < keep_prob).float()
            sequence = sequence * mask
        
        return sequence
    
    def forward(self, t, x, wa, wr, we, prev_x=None, prev_wa=None, train=True, **kwargs) -> torch.Tensor:
        """
        Forward pass with robust batch handling.
        
        Args:
            t: timestep, scalar or (B,) tensor
            x: current frames (B, L, dim_w)
            wa: audio features (B, L, dim_a)
            wr: reference features (B, dim_w)
            we: emotion features (B, L, dim_e)
            prev_x: previous frames (B, num_prev, dim_w) or None
            prev_wa: previous audio (B, num_prev, dim_a) or None
            train: training mode flag
        """
        # 1. Get batch size and device
        B = x.shape[0]
        device = x.device
        
        # 2. Handle timestep - ensure it's properly shaped
        if isinstance(t, (int, float)):
            t = torch.tensor(t, device=device)
        
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        if t.shape[0] == 1 and B > 1:
            t = t.expand(B)
        elif t.shape[0] != B:
            raise ValueError(f"Timestep batch size {t.shape[0]} doesn't match input batch size {B}")
        
        # 3. Time embedding
        t_emb = self.t_embedder(t)  # (B, hidden_size)
        if t_emb.dim() == 2:
            t_emb = t_emb.unsqueeze(1)  # (B, 1, hidden_size)
        
        # 4. Apply conditional dropout (for classifier-free guidance)
        wa = self.sequence_embedder(wa, dropout_prob=self.opt.audio_dropout_prob, train=train)
        wr = self.sequence_embedder(wr.unsqueeze(1) if wr.dim() == 2 else wr, 
                                     dropout_prob=self.opt.ref_dropout_prob, train=train)
        we = self.sequence_embedder(we, dropout_prob=self.opt.emotion_dropout_prob, train=train)
        
        # 5. Handle autoregressive concatenation
        if prev_x is not None and prev_wa is not None:
            prev_x = self.sequence_embedder(prev_x, dropout_prob=0.5, train=train)
            prev_wa = self.sequence_embedder(prev_wa, dropout_prob=0.5, train=train)
            
            x = torch.cat([prev_x, x], dim=1)
            wa = torch.cat([prev_wa, wa], dim=1)
        
        # 6. Embed main sequence
        x = self.x_embedder(x)
        
        # 7. Add positional embeddings
        seq_len = x.shape[1]
        pos_embed = self.pos_embed[:, :seq_len, :].expand(B, -1, -1)
        x = x + pos_embed
        
        # 8. Prepare conditions
        # Expand reference to match sequence length
        if wr.dim() == 3 and wr.shape[1] == 1:
            wr = wr.expand(-1, seq_len, -1)
        elif wr.dim() == 2:
            wr = wr.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Expand emotion if needed
        if we.shape[1] != seq_len:
            # Handle case where emotion might be single frame
            if we.shape[1] == 1:
                we = we.expand(-1, seq_len, -1)
            else:
                raise ValueError(f"Emotion sequence length {we.shape[1]} doesn't match {seq_len}")
        
        # Concatenate conditions
        c = torch.cat([wr, wa, we], dim=-1)
        c = self.c_embedder(c)
        c = c + t_emb  # Add time embedding (broadcasts automatically)
        
        # 9. Get attention mask
        if seq_len != self.alignment_mask.shape[0]:
            # Create mask for current sequence length
            current_mask = enc_dec_mask(seq_len, seq_len, 1, expansion=self.opt.attention_window)
            current_mask = current_mask.to(device)
        else:
            current_mask = self.alignment_mask
        
        # 10. Forward through transformer blocks
        for block in self.blocks:
            x = block(x, c, current_mask)
        
        # 11. Decode
        return self.decoder(x, c)
    
    @torch.no_grad()
    def forward_with_cfg(self, t, x, wa, wr, we, prev_x, prev_wa, 
                         a_cfg_scale=1.0, r_cfg_scale=1.0, e_cfg_scale=1.0, **kwargs) -> torch.Tensor:
        """
        Classifier-Free Guidance inference with full batch support.
        
        Supports arbitrary batch sizes and handles CFG properly.
        """
        B = x.shape[0]
        device = x.device
        
        # Check if CFG is needed
        use_cfg = (a_cfg_scale != 1.0 or e_cfg_scale != 1.0)
        
        if use_cfg:
            # Create null conditions
            null_wa = torch.zeros_like(wa)
            null_we = torch.zeros_like(we)
            
            # Concatenate for three forward passes:
            # 0: Unconditional (null audio, null emotion)
            # 1: Full conditional (real audio, real emotion)
            # 2: Audio-only (real audio, null emotion)
            audio_cat = torch.cat([null_wa, wa, wa], dim=0)  # (3B, L, dim_a)
            emotion_cat = torch.cat([null_we, we, null_we], dim=0)  # (3B, L, dim_e)
            ref_cat = torch.cat([wr, wr, wr], dim=0)  # (3B, dim_w)
            x_cat = torch.cat([x, x, x], dim=0)  # (3B, L, dim_w)
            
            if prev_x is not None and prev_wa is not None:
                prev_x_cat = torch.cat([prev_x, prev_x, prev_x], dim=0)
                prev_wa_cat = torch.cat([prev_wa, prev_wa, prev_wa], dim=0)
            else:
                prev_x_cat = None
                prev_wa_cat = None
            
            # Forward pass with concatenated batch
            # t will be automatically expanded to 3B in forward()
            model_output = self.forward(
                t, x_cat, audio_cat, ref_cat, emotion_cat, 
                prev_x_cat, prev_wa_cat, train=False
            )
            
            # Split outputs
            uncond, full_cond, audio_only = torch.chunk(model_output, 3, dim=0)
            
            # Apply CFG formula:
            # out = uncond + audio_scale * (audio_only - uncond) + emotion_scale * (full_cond - audio_only)
            out = uncond + a_cfg_scale * (audio_only - uncond) + e_cfg_scale * (full_cond - audio_only)
            
            return out
        else:
            # Standard forward without CFG
            return self.forward(t, x, wa, wr, we, prev_x, prev_wa, train=False)
    
    def generate(self, x_init, wa, wr, we, prev_x=None, prev_wa=None, 
                 num_steps=50, cfg_scale_audio=1.0, cfg_scale_emotion=1.0, 
                 method='euler', **kwargs):
        """
        Generate samples using ODE integration with batching support.
        
        Args:
            x_init: initial noise (B, L, dim_w)
            wa: audio features (B, L, dim_a)
            wr: reference features (B, dim_w)
            we: emotion features (B, L, dim_e)
            num_steps: number of integration steps
            cfg_scale_audio: classifier-free guidance scale for audio
            cfg_scale_emotion: classifier-free guidance scale for emotion
            method: ODE solver method
        """
        device = x_init.device
        
        # Define ODE function
        def ode_func(t, x):
            # t is scalar, x is (B, L, D)
            return self.forward_with_cfg(
                t, x, wa, wr, we, prev_x, prev_wa,
                a_cfg_scale=cfg_scale_audio,
                e_cfg_scale=cfg_scale_emotion
            )
        
        # Time schedule
        t_span = torch.linspace(0, 1, num_steps, device=device)
        
        # Integrate ODE
        trajectory = odeint(ode_func, x_init, t_span, method=method)
        
        # Return final state
        return trajectory[-1]