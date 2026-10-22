from fast_float.models import BaseModel

from .encoder import Encoder
from .styledecoder import Synthesis


class Generator(BaseModel):
    def __init__(
        self,
        size,
        style_dim=512,
        motion_dim=20,
        channel_multiplier=1,
        blur_kernel=None,
    ):
        if blur_kernel is None:
            blur_kernel = [1, 3, 3, 1]
        super().__init__()

        self.enc = Encoder(size, style_dim, motion_dim)
        self.dec = Synthesis(size, style_dim, motion_dim, blur_kernel, channel_multiplier)

    def get_direction(self):
        return self.dec.direction(None)

    def synthesis(self, wa, alpha, feat):
        img, _flow = self.dec(wa, alpha, feat)
        return img

    def forward(self, img_source, img_drive, h_start=None):
        wa, alpha, feats = self.enc(img_source, img_drive, h_start)
        img_recon, flow = self.dec(wa, alpha, feats)
        return {"d_hat": img_recon, "flow": flow}

    def to_infer(self, fold_blur: bool = False, dtype=None, channels_last: bool = False):
        """Convert generator's decoder to optimized SynthesisInfer."""
        if hasattr(self.dec, "to_infer"):
            self.dec = self.dec.to_infer(fold_blur=fold_blur, dtype=dtype, channels_last=channels_last)
        return self
