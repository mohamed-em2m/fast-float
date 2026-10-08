# Batched Latent Decoding & GPU Auto-Sizing in FLOAT

## 1. Overview

During speech-driven facial animation with **FLOAT**, generating long sequences can produce hundreds or thousands of latent frames. Decoding all frames in a single forward pass causes GPU Out-Of-Memory (**OOM**) errors, while decoding frames one-by-one underutilizes GPU Tensor Cores and slows down generation.

To solve this, FLOAT includes **Micro-Batch Frame Gathering & Decoding** with **Dynamic GPU VRAM Auto-Sizing**:
1. **Gathering in Micro-Batches**: Latents are partitioned into chunks (e.g., gathering 4 frames together) and decoded through `Generator.dec` in parallel.
2. **Dynamic GPU Sizing**: Automatically detects free and total VRAM on the target CUDA device to choose the optimal batch size without manual configuration.
3. **Zero-Copy Feature Alignment**: Slices and broadcasts multi-resolution identity features (`s_r_feats`) without redundant memory copies.
4. **Dual 2D and 3D Latent Support**: Seamlessly processes both flattened 2D sequence latents $[N, D]$ and batched 3D multi-identity latents $[B, T, D]$.

---

## 2. Memory Tier Auto-Sizing Logic

The method [`FLOAT.get_optimal_decode_batch_size(device)`](file:///c:/Users/emam2/tech_projects/float-fast/fast_float/models/float/FLOAT.py) inspects the target device using `torch.cuda.mem_get_info()`:

| Total VRAM Tier | Free VRAM Threshold | Selected Batch Size | Typical Hardware |
|---|---|:---:|---|
| **Ultra ($\ge 22\text{ GB}$)** | $\ge 8\text{ GB}$ free | **16** | NVIDIA RTX 3090, RTX 4090, A100, H100 |
| **High ($11 - 22\text{ GB}$)** | $\ge 4\text{ GB}$ free | **8** | NVIDIA RTX 3060 12GB, RTX 4070, RTX 4080 |
| **Standard ($6 - 11\text{ GB}$)** | $\ge 2\text{ GB}$ free | **4** *(Default)* | NVIDIA RTX 2060, RTX 3060 8GB, RTX 4060 |
| **Constrained ($< 6\text{ GB}$)** | $\ge 1\text{ GB}$ free | **2** | GTX 1650, GTX 1660, Mobile GPUs |
| **Critical / CPU** | $< 1\text{ GB}$ free or CPU | **1 or 4 (CPU)** | Fallback / CPU execution |

> [!NOTE]
> If `opt.decode_batch_size` is explicitly configured in `self.opt`, it takes absolute precedence over automatic detection.

---

## 3. Core Architecture

### 3.1 Feature Alignment (`_decode_latent_batched`)
The decoder requires feature maps (`s_r_feats`) matching the batch size of the latent input:
- **Single reference image ($f.\text{shape}[0] == 1$)**: Features are expanded dynamically to chunk size $M$ (`f.expand(M, *dims)`), which is a zero-copy strided view.
- **Per-frame reference features ($f.\text{shape}[0] == N$)**: Features are directly sliced per chunk (`f[i : i + M]`).
- **Batched identities ($B$ references, $T$ frames each)**: Features are aligned along the identity dimension and sliced across the chunk window.

### 3.2 2D & 3D Latent Processing

```
3D Latents [B, T, D]                2D Latents [N, D]
        │                                   │
        ▼                                   ▼
Flatten to [B*T, D]                 Direct length N
        │                                   │
        └───────────────┬───────────────────┘
                        ▼
            Check N against batch_size (bs)
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
     N <= bs                        N > bs
  (Single Pass)            Chunk into [0:bs], [bs:2*bs], ...
         │                             │
         │                     Parallel dec() per chunk
         │                             │
         │                     torch.cat(chunks, dim=0)
         └──────────────┬──────────────┘
                        ▼
       Reshape to original shape:
       - 3D: [B, T, 3, H, W]
       - 2D: [N, 3, H, W]
```

---

## 4. API Usage Examples

### Direct Decoding via `decode_latent_into_image`
```python
import torch
from fast_float.models.float.FLOAT import FLOAT

# Initialize model
model = FLOAT(opt)

# Extract reference features from identity image
s_r, r_s_lambda, s_r_feats = model.encode_image_into_latent(reference_image)

# Latent trajectory [N=100, D=512]
latents = torch.randn(100, 512, device="cuda")

# 1. Automatic GPU-adapted batch size (e.g., 4 on 8GB GPU, 8 on 12GB GPU):
frames = model.decode_latent_into_image(
    s_r=s_r,
    s_r_feats=s_r_feats,
    r_d=latents,
    batch_size=None,  # Automatically detects available VRAM
)

# 2. Explicitly gathering each 4 frames together:
frames = model.decode_latent_into_image(
    s_r=s_r,
    s_r_feats=s_r_feats,
    r_d=latents,
    batch_size=4,
)
```

### Decoding Batched Multi-Identity Latents (3D)
```python
# Latents for B=2 identities across T=25 frames each [2, 25, 512]
latents_3d = torch.randn(2, 25, 512, device="cuda")

# Decodes 50 total frames gathered in micro-batches of 4:
video_tensors = model.decode_latent_into_image(
    s_r=None,
    s_r_feats=batched_s_r_feats,
    r_d=latents_3d,
    batch_size=4,
)
# Output shape: [2, 25, 3, 256, 256]
```

### Configuring via Options
```python
opt.decode_batch_size = 4  # Forces 4-frame micro-batch gathering in both sample and inference
```

---

## 5. Verification Suite

Run the batch processing test suite to verify auto-sizing, gathering, and broadcasting:

```powershell
python "scratch/test_batch_processing.py"
```

Verified test results:
- `[PASS] FMT forward multi-batch (B=4)`
- `[PASS] Synthesis batched frames (B=4)`
- `[PASS] Auto decode batch size selected based on device: 4`
- `[PASS] User override decode_batch_size=4 verified`
- `[PASS] Batched decode (9 frames chunked into 4, 4, 1) passed`
- `[PASS] 3D [B=2, T=5] batched decode (10 frames chunked into 4, 4, 2) passed`
