from torch import nn
from .encoder import Encoder
from .styledecoder import Synthesis
from models import BaseModel


class Generator(BaseModel):
    def __init__(self, size, style_dim=512, motion_dim=20, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.enc = Encoder(size, style_dim, motion_dim)
        self.dec = Synthesis(size, style_dim, motion_dim, blur_kernel, channel_multiplier)
        self.size = size
        self.style_dim = style_dim
        self.motion_dim = motion_dim
    
    def get_direction(self):
        return self.dec.direction(None)
    
    def synthesis(self, wa, alpha, feat):
        """
        Synthesis with batching support.
        
        Args:
            wa: latent codes [batch, style_dim] or [batch, n_frames, style_dim]
            alpha: motion parameters [batch, motion_dim] or list of alpha tensors
            feat: feature maps, list of tensors with batch dimension
        """
        # Handle temporal batching if wa has 3 dimensions
        if wa.dim() == 3:
            batch, n_frames, _ = wa.shape
            imgs = []
            flows = []
            for t in range(n_frames):
                # Process each frame with appropriate alpha
                if isinstance(alpha, list):
                    alpha_t = [a[:, t] if a.dim() == 3 else a for a in alpha]
                else:
                    alpha_t = alpha[:, t] if alpha.dim() == 3 else alpha
                
                feat_t = [f[:, t] if f.dim() == 5 else f for f in feat]
                img, flow = self.dec(wa[:, t], alpha_t, feat_t)
                imgs.append(img)
                flows.append(flow)
            
            img = torch.stack(imgs, dim=1)  # [batch, n_frames, C, H, W]
            flow = torch.stack(flows, dim=1)  # [batch, n_frames, 2, H, W]
        else:
            img, flow = self.dec(wa, alpha, feat)
        
        return img
    
    def forward(self, img_source, img_drive, h_start=None):
        """
        Forward pass with batching support.
        
        Args:
            img_source: source images [batch, C, H, W] or [batch, n_frames, C, H, W]
            img_drive: driving images [batch, C, H, W] or [batch, n_frames, C, H, W]
            h_start: optional starting hidden state [batch, style_dim]
        
        Returns:
            dict with 'd_hat' (reconstructed images) and 'flow' (optical flow)
        """
        # Handle temporal sequences
        if img_source.dim() == 5:
            batch, n_frames, C, H, W = img_source.shape
            
            # Reshape to process all frames in batch
            img_source_flat = img_source.view(batch * n_frames, C, H, W)
            img_drive_flat = img_drive.view(batch * n_frames, C, H, W)
            
            # Expand h_start for all frames if provided
            if h_start is not None:
                h_start_flat = h_start.unsqueeze(1).repeat(1, n_frames, 1).view(batch * n_frames, -1)
            else:
                h_start_flat = None
            
            # Encode
            wa, alpha, feats = self.enc(img_source_flat, img_drive_flat, h_start_flat)
            
            # Reshape back to temporal structure
            wa = wa.view(batch, n_frames, -1)
            if isinstance(alpha, list):
                alpha = [a.view(batch, n_frames, -1) if a is not None else None for a in alpha]
            else:
                alpha = alpha.view(batch, n_frames, -1) if alpha is not None else None
            
            # Reshape features
            feats_reshaped = []
            for feat in feats:
                if feat.dim() == 4:
                    _, C_f, H_f, W_f = feat.shape
                    feats_reshaped.append(feat.view(batch, n_frames, C_f, H_f, W_f))
                else:
                    feats_reshaped.append(feat)
            
            # Decode frame by frame
            imgs_recon = []
            flows = []
            for t in range(n_frames):
                wa_t = wa[:, t]
                if isinstance(alpha, list):
                    alpha_t = [a[:, t] if a is not None and a.dim() == 3 else a for a in alpha]
                else:
                    alpha_t = alpha[:, t] if alpha is not None and alpha.dim() == 3 else alpha
                
                feat_t = [f[:, t] if f.dim() == 5 else f for f in feats_reshaped]
                
                img_recon_t, flow_t = self.dec(wa_t, alpha_t, feat_t)
                imgs_recon.append(img_recon_t)
                flows.append(flow_t)
            
            img_recon = torch.stack(imgs_recon, dim=1)  # [batch, n_frames, C, H, W]
            flow = torch.stack(flows, dim=1)  # [batch, n_frames, 2, H, W]
            
        else:
            # Single frame processing (original behavior)
            wa, alpha, feats = self.enc(img_source, img_drive, h_start)
            img_recon, flow = self.dec(wa, alpha, feats)
        
        return {'d_hat': img_recon, 'flow': flow}
    
    def forward_batch(self, img_source, img_drive, h_start=None, chunk_size=None):
        """
        Memory-efficient forward pass for large batches.
        
        Args:
            img_source: source images [batch, C, H, W] or [batch, n_frames, C, H, W]
            img_drive: driving images [batch, C, H, W] or [batch, n_frames, C, H, W]
            h_start: optional starting hidden state [batch, style_dim]
            chunk_size: process batch in chunks to save memory
        
        Returns:
            dict with 'd_hat' (reconstructed images) and 'flow' (optical flow)
        """
        if chunk_size is None:
            return self.forward(img_source, img_drive, h_start)
        
        batch_size = img_source.shape[0]
        device = img_source.device
        
        all_img_recon = []
        all_flow = []
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            
            img_source_chunk = img_source[i:end_idx]
            img_drive_chunk = img_drive[i:end_idx]
            h_start_chunk = h_start[i:end_idx] if h_start is not None else None
            
            with torch.cuda.amp.autocast(enabled=False):  # Can enable mixed precision if needed
                result = self.forward(img_source_chunk, img_drive_chunk, h_start_chunk)
            
            all_img_recon.append(result['d_hat'])
            all_flow.append(result['flow'])
        
        return {
            'd_hat': torch.cat(all_img_recon, dim=0),
            'flow': torch.cat(all_flow, dim=0)
        }