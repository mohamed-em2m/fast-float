import os, torch, random, cv2, torchvision, librosa, datetime, tempfile, face_alignment
import numpy as np
import albumentations as A
import albumentations.pytorch.transforms as A_pytorch
from tqdm import tqdm
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor
from models.float.FLOAT import FLOAT
from options.base_options import BaseOptions
import time
import uuid
import logging
import struct
import wave
import subprocess
import shutil
from typing import List, Union, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Setup a logger with console output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logger(__name__, level=logging.INFO)


class SimpleVideoEncoder:
    """
    A from-scratch video encoder that creates AVI files with Motion JPEG encoding.
    This is a simple, pure-Python implementation that doesn't rely on ffmpeg.
    """
    
    def __init__(self, width: int, height: int, fps: int, output_path: str):
        self.width = width
        self.height = height
        self.fps = fps
        self.output_path = output_path
        self.frame_count = 0
        self.frames = []
        
    def add_frame(self, frame: np.ndarray):
        """Add a frame to the video. Frame should be RGB uint8 numpy array (H, W, 3)"""
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Frame shape {frame.shape} doesn't match video size ({self.height}, {self.width}, 3)")
        
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.frames.append(bgr_frame)
        self.frame_count += 1
    
    def write_avi_with_audio(self, audio_path: str = None):
        """Write frames to AVI file using OpenCV VideoWriter"""
        if self.frame_count == 0:
            raise ValueError("No frames to write!")
        
        # Use MJPEG codec (widely supported, no external dependencies)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        
        # Create temporary video file without audio
        temp_video = self.output_path.replace('.mp4', '_temp.avi').replace('.avi', '_temp.avi')
        
        out = cv2.VideoWriter(temp_video, fourcc, self.fps, (self.width, self.height))
        
        if not out.isOpened():
            raise RuntimeError("Failed to open video writer")
        
        logger.info(f"Writing {self.frame_count} frames to {temp_video}...")
        for frame in tqdm(self.frames, desc="Encoding video"):
            out.write(frame)
        
        out.release()
        logger.info(f"✅ Video encoding complete: {temp_video}")
        
        # If audio is provided, combine using OpenCV (limited) or just save video-only
        if audio_path and os.path.exists(audio_path):
            logger.warning("Audio merging without ffmpeg is limited. Saving video-only version.")
            logger.info(f"To add audio manually, use: ffmpeg -i {temp_video} -i {audio_path} -c:v copy -c:a aac {self.output_path}")
        
        # Rename temp to final output
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        os.rename(temp_video, self.output_path)
        
        return self.output_path
    
    def write(self):
        """Write frames without audio"""
        return self.write_avi_with_audio(audio_path=None)


class DataProcessor:
    def __init__(self, opt):
        self.opt = opt
        self.fps = opt.fps
        self.sampling_rate = opt.sampling_rate
        self.input_size = opt.input_size
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
        
        if os.path.exists(opt.wav2vec_model_path):
            self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
                opt.wav2vec_model_path, 
                local_files_only=True
            )
        else:
            self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(opt.wav2vec_model_path)
        
        self.transform = A.Compose([
            A.Resize(height=opt.input_size, width=opt.input_size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            A_pytorch.ToTensorV2(),
        ])
    
    @torch.no_grad()
    def process_img(self, img: np.ndarray) -> np.ndarray:
        """Process and crop face from image"""
        mult = 360. / img.shape[0]
        resized_img = cv2.resize(
            img, 
            dsize=(0, 0), 
            fx=mult, 
            fy=mult, 
            interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC
        )
        bboxes = self.fa.face_detector.detect_from_image(resized_img)
        bboxes = [
            (int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), score) 
            for (x1, y1, x2, y2, score) in bboxes if score > 0.95
        ]
        
        if not bboxes:
            raise ValueError("No face detected with confidence > 0.95")
        
        bboxes = bboxes[0]
        
        bsy = int((bboxes[3] - bboxes[1]) / 2)
        bsx = int((bboxes[2] - bboxes[0]) / 2)
        my = int((bboxes[1] + bboxes[3]) / 2)
        mx = int((bboxes[0] + bboxes[2]) / 2)
        bs = int(max(bsy, bsx) * 1.6)
        
        img = cv2.copyMakeBorder(img, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=0)
        my, mx = my + bs, mx + bs
        crop_img = img[my - bs:my + bs, mx - bs:mx + bs]
        crop_img = cv2.resize(
            crop_img, 
            dsize=(self.input_size, self.input_size), 
            interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC
        )
        return crop_img
    
    def default_img_loader(self, path: str) -> np.ndarray:
        """Load image from path"""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image from {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def default_aud_loader(self, path: str) -> torch.Tensor:
        """Load audio from path"""
        speech_array, sampling_rate = librosa.load(path, sr=self.sampling_rate)
        return self.wav2vec_preprocessor(
            speech_array, 
            sampling_rate=sampling_rate, 
            return_tensors='pt'
        ).input_values[0]
    
    def preprocess(self, ref_path: str, audio_path: str, no_crop: bool = False) -> dict:
        """Preprocess single image-audio pair"""
        s = self.default_img_loader(ref_path)
        if not no_crop:
            s = self.process_img(s)
        s = self.transform(image=s)['image'].unsqueeze(0)
        a = self.default_aud_loader(audio_path).unsqueeze(0)
        return {'s': s, 'a': a, 'p': None, 'e': None}
    
    def preprocess_batch(
        self, 
        ref_paths: List[str], 
        audio_paths: List[str], 
        no_crop: bool = False,
        num_workers: int = 4
    ) -> dict:
        """
        Preprocess multiple image-audio pairs in parallel.
        
        Args:
            ref_paths: List of reference image paths
            audio_paths: List of audio file paths
            no_crop: Whether to skip face cropping
            num_workers: Number of parallel workers for preprocessing
        
        Returns:
            dict with batched tensors
        """
        if len(ref_paths) != len(audio_paths):
            raise ValueError("Number of images and audio files must match")
        
        # Process in parallel using ThreadPoolExecutor
        def process_single(ref_path, audio_path):
            return self.preprocess(ref_path, audio_path, no_crop)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_single, ref_paths, audio_paths))
        
        # Stack results into batches
        s_batch = torch.cat([r['s'] for r in results], dim=0)
        a_batch = torch.cat([r['a'] for r in results], dim=0)
        
        return {'s': s_batch, 'a': a_batch, 'p': None, 'e': None}


class OptimizedInferenceAgent:
    def __init__(self, opt):
        torch.cuda.empty_cache()
        self.opt = opt
        self.rank = opt.rank
        self.optimization_mode = opt.optimization_mode if hasattr(opt, 'optimization_mode') else 'torch_compile'
        self.use_fp16 = opt.use_fp16 if hasattr(opt, 'use_fp16') else False
        
        # Load Model
        self.load_model()
        self.load_weight(opt.ckpt_path, rank=self.rank)
        self.G.to(self.rank)
        self.G.eval()
        
        # Apply optimization
        self.apply_optimization()
        
        # Load Data Processor
        self.data_processor = DataProcessor(opt)
        
        print(f"✓ Model loaded with {self.optimization_mode} optimization")
    
    def load_model(self) -> None:
        self.G = FLOAT(self.opt)
    
    def load_weight(self, checkpoint_path: str, rank: int) -> None:
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        with torch.no_grad():
            for model_name, model_param in self.G.named_parameters():
                if model_name in state_dict:
                    model_param.copy_(state_dict[model_name].to(rank))
                elif "wav2vec2" in model_name:
                    pass
                else:
                    print(f"! Warning; {model_name} not found in state_dict.")
        del state_dict
    
    def apply_optimization(self):
        """Apply various optimization strategies"""
        
        if self.optimization_mode == 'torch_compile':
            try:
                if hasattr(torch, 'compile'):
                    print("Applying torch.compile optimization...")
                    self.G_opt = torch.compile(
                        self.G,
                        mode='max-autotune',
                        fullgraph=False
                    )
                    print("✓ torch.compile applied successfully")
                else:
                    print("! torch.compile not available (requires PyTorch 2.0+)")
                    self.G_opt = self.G
            except Exception as e:
                print(f"! torch.compile failed: {e}")
                self.G_opt = self.G
        
        elif self.optimization_mode == 'channels_last':
            print("Applying channels_last memory format...")
            try:
                for module in self.G.modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
                        module = module.to(memory_format=torch.channels_last)
                self.G_opt = self.G
                print("✓ Channels last format applied")
            except Exception as e:
                print(f"! Channels last conversion failed: {e}, skipping...")
                self.G_opt = self.G
        
        elif self.optimization_mode == 'amp':
            print("Using Automatic Mixed Precision (AMP)...")
            self.G_opt = self.G
            self.use_amp = True
            print("✓ AMP enabled")
        
        elif self.optimization_mode == 'cudnn_benchmark':
            print("Enabling cuDNN benchmark mode...")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            self.G_opt = self.G
            print("✓ cuDNN benchmark enabled")
        
        elif self.optimization_mode == 'all':
            print("Applying all optimizations...")
            
            try:
                for module in self.G.modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
                        module = module.to(memory_format=torch.channels_last)
                print("✓ Channels last applied to conv layers")
            except Exception as e:
                print(f"! Channels last skipped: {e}")
            
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            print("✓ cuDNN benchmark enabled")
            
            if hasattr(torch, 'compile'):
                try:
                    self.G_opt = torch.compile(
                        self.G,
                        mode='max-autotune',
                        fullgraph=False
                    )
                    print("✓ All optimizations applied")
                except Exception as e:
                    print(f"! torch.compile failed: {e}")
                    self.G_opt = self.G
                    print("✓ Partial optimizations applied")
            else:
                self.G_opt = self.G
                print("✓ Partial optimizations applied")
        
        else:
            print("No optimization applied")
            self.G_opt = self.G
        
        if self.use_fp16:
            print("Enabling FP16 inference...")
            self.use_amp_fp16 = True
            print("✓ FP16 will be applied via autocast during inference")
    
    def save_video(self, vid_target_recon: torch.Tensor, video_path: str, audio_path: str = None) -> str:
        """
        Save video with proper CUDA synchronization to prevent corruption.
        Supports both single videos and batched videos.
        
        Args:
            vid_target_recon: Video tensor (T, C, H, W) or (B, T, C, H, W)
            video_path: Output path (for single video) or directory (for batch)
            audio_path: Optional audio file path
        """
        start_time = time.time()

        # Sanity: ffmpeg present
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found in PATH")

        # Full GPU synchronization
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            time.sleep(0.05)
        
        # Handle batched input
        is_batch = vid_target_recon.ndim == 5
        
        if is_batch:
            # Process batch of videos
            B = vid_target_recon.shape[0]
            logger.info(f"Saving batch of {B} videos...")
            
            # Ensure video_path is a directory for batch
            if os.path.isfile(video_path):
                video_dir = os.path.dirname(video_path)
                base_name = os.path.splitext(os.path.basename(video_path))[0]
            else:
                video_dir = video_path
                base_name = "video"
            
            os.makedirs(video_dir, exist_ok=True)
            
            saved_paths = []
            for i in range(B):
                video_i = vid_target_recon[i]  # (T, C, H, W)
                output_path_i = os.path.join(video_dir, f"{base_name}_{i:04d}.mp4")
                saved_path = self._save_single_video(video_i, output_path_i, audio_path)
                saved_paths.append(saved_path)
            
            total_time = time.time() - start_time
            logger.info(f"✅ Saved {B} videos in {total_time:.3f}s -> {video_dir}")
            return saved_paths
        else:
            # Single video
            return self._save_single_video(vid_target_recon, video_path, audio_path)
    
    def _save_single_video(self, vid_t: torch.Tensor, video_path: str, audio_path: str = None) -> str:
        """Save a single video tensor to file"""
        start_time = time.time()
        
        # Prepare tensor -> CPU uint8 numpy
        vid_t = vid_t.detach()
        
        # Blocking CPU transfer
        if vid_t.is_cuda:
            vid_t = vid_t.to("cpu", non_blocking=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Expect shape (T, C, H, W)
        if vid_t.ndim != 4:
            raise ValueError("vid_target_recon must be a 4D tensor (T, C, H, W)")
        
        # Permute to (T, H, W, C) and ensure contiguous
        vid_t = vid_t.permute(0, 2, 3, 1).contiguous()
        
        # Convert float in [-1,1] => uint8 [0,255]
        if torch.is_floating_point(vid_t):
            vid_t = (vid_t.clamp(-1.0, 1.0) * 127.5 + 127.5).to(torch.uint8)
        else:
            vid_t = vid_t.to(torch.uint8)
        
        # Convert to numpy
        vid_np = vid_t.numpy()
        T, H, W, C = vid_np.shape
        
        # Verify numpy array is valid
        if not vid_np.flags['C_CONTIGUOUS']:
            vid_np = np.ascontiguousarray(vid_np)
        
        if C not in (3, 4):
            raise ValueError(f"Expected 3 or 4 channels, got {C}")
        
        fps = int(getattr(self.opt, "fps", 25))
        
        # Build ffmpeg command
        pix_fmt_in = "rgba" if C == 4 else "rgb24"
        size_str = f"{W}x{H}"
        
        base_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", pix_fmt_in,
            "-s", size_str,
            "-r", str(fps),
            "-i", "-",
        ]
        
        vcodec = "h264_nvenc"
        encode_opts = [
            "-c:v", vcodec,
            "-preset", "p1",
            "-tune", "hq",
            "-rc", "constqp",
            "-qp", "23",
            "-delay", "0",
            "-no-scenecut", "1",
            "-forced-idr", "1",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]
        
        cmd = base_cmd + encode_opts + [video_path]
        
        # Execute with retry logic
        max_retries = 2
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    bufsize=65536
                )
                
                # Write frames
                chunk_size = 25
                for chunk_start in range(0, T, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, T)
                    
                    for i in range(chunk_start, chunk_end):
                        frame = vid_np[i]
                        frame_bytes = frame.tobytes()
                        proc.stdin.write(frame_bytes)
                    
                    if retry_count == 0 and chunk_start == 0:
                        time.sleep(0.01)
                
                proc.stdin.close()
                ret = proc.wait(timeout=300)
                
                if ret != 0:
                    stderr = proc.stderr.read().decode(errors="ignore")
                    if "CUDA_ERROR" in stderr or "initialization error" in stderr:
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.warning(f"⚠️ NVENC init error, retry {retry_count}/{max_retries}")
                            time.sleep(0.5)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            continue
                    raise RuntimeError(f"ffmpeg exited with code {ret}: {stderr}")
                
                break
            
            except Exception as e:
                if proc:
                    try:
                        proc.kill()
                    except:
                        pass
                
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"❌ Encoding failed after {max_retries} retries: {e}")
                    raise
                
                logger.warning(f"⚠️ Encoding attempt {retry_count} failed, retrying...")
                time.sleep(0.5)
        
        total_time = time.time() - start_time
        logger.info(f"✅ Video saved: {total_time:.3f}s -> {video_path}")
        return video_path
    
    @torch.no_grad()
    def run_inference(
        self,
        res_video_path: str,
        ref_path: Union[str, List[str]],
        audio_path: Union[str, List[str]],
        a_cfg_scale: float = 2.0,
        r_cfg_scale: float = 1.0,
        e_cfg_scale: float = 1.0,
        emo: str = 'S2E',
        nfe: int = 10,
        no_crop: bool = False,
        seed: int = 25,
        verbose: bool = True,
        benchmark: bool = False,
        batch_size: int = None,
    ) -> Union[str, List[str]]:
        """
        Run inference with support for single or batch processing.
        
        Args:
            res_video_path: Output path or directory
            ref_path: Single path or list of reference image paths
            audio_path: Single path or list of audio paths
            batch_size: Maximum batch size for memory efficiency (None = process all at once)
            
        Returns:
            Output video path(s)
        """
        # Detect batch vs single
        is_batch = isinstance(ref_path, list)
        
        if is_batch:
            if not isinstance(audio_path, list):
                raise ValueError("If ref_path is a list, audio_path must also be a list")
            return self._run_batch_inference(
                res_video_path, ref_path, audio_path,
                a_cfg_scale, r_cfg_scale, e_cfg_scale,
                emo, nfe, no_crop, seed, verbose, batch_size
            )
        else:
            return self._run_single_inference(
                res_video_path, ref_path, audio_path,
                a_cfg_scale, r_cfg_scale, e_cfg_scale,
                emo, nfe, no_crop, seed, verbose, benchmark
            )
    
    @torch.no_grad()
    def _run_single_inference(
        self,
        res_video_path: str,
        ref_path: str,
        audio_path: str,
        a_cfg_scale: float,
        r_cfg_scale: float,
        e_cfg_scale: float,
        emo: str,
        nfe: int,
        no_crop: bool,
        seed: int,
        verbose: bool,
        benchmark: bool
    ) -> str:
        """Single inference (original implementation)"""
        
        timings = {}
        total_start = time.time()
        
        # Preprocess
        preprocess_start = time.time()
        data = self.data_processor.preprocess(ref_path, audio_path, no_crop=no_crop)
        timings['preprocess'] = time.time() - preprocess_start
        
        if verbose:
            print(f"> [Done] Preprocess ({timings['preprocess']:.3f}s)")
        
        # Benchmark mode
        if benchmark:
            self._run_benchmark(data, a_cfg_scale, r_cfg_scale, e_cfg_scale, emo, nfe, seed)
        
        # Inference
        use_amp = hasattr(self, 'use_amp') and self.use_amp
        use_amp_fp16 = hasattr(self, 'use_amp_fp16') and self.use_amp_fp16
        
        inference_start = time.time()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        if use_amp or use_amp_fp16:
            with torch.cuda.amp.autocast(dtype=torch.float16 if use_amp_fp16 else torch.float32):
                d_hat = self.G_opt.inference(
                    data=data,
                    a_cfg_scale=a_cfg_scale,
                    r_cfg_scale=r_cfg_scale,
                    e_cfg_scale=e_cfg_scale,
                    emo=emo,
                    nfe=nfe,
                    seed=seed
                )['d_hat']
        else:
            d_hat = self.G_opt.inference(
                data=data,
                a_cfg_scale=a_cfg_scale,
                r_cfg_scale=r_cfg_scale,
                e_cfg_scale=e_cfg_scale,
                emo=emo,
                nfe=nfe,
                seed=seed
            )['d_hat']
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        timings['inference'] = time.time() - inference_start
        
        if verbose:
            print(f"> [Done] Inference ({timings['inference']:.3f}s)")
        
        # Save
        save_start = time.time()
        res_video_path = self.save_video(d_hat.float(), res_video_path, audio_path)
        timings['save'] = time.time() - save_start
        
        timings['total'] = time.time() - total_start
        
        if verbose:
            self._print_timings(timings)
            print(f"> Result saved at {res_video_path}")
        
        return res_video_path
    
    @torch.no_grad()
    def _run_batch_inference(
        self,
        res_video_path: str,
        ref_paths: List[str],
        audio_paths: List[str],
        a_cfg_scale: float,
        r_cfg_scale: float,
        e_cfg_scale: float,
        emo: str,
        nfe: int,
        no_crop: bool,
        seed: int,
        verbose: bool,
        batch_size: Optional[int]
    ) -> List[str]:
        """Batch inference for multiple inputs"""
        
        total_samples = len(ref_paths)
        logger.info(f"Processing batch of {total_samples} samples...")
        
        timings = {'preprocess': 0, 'inference': 0, 'save': 0}
        total_start = time.time()
        
        # Determine effective batch size
        if batch_size is None:
            batch_size = total_samples
        
        all_results = []
        
        # Process in chunks
        for chunk_start in range(0, total_samples, batch_size):
            chunk_end = min(chunk_start + batch_size, total_samples)
            chunk_refs = ref_paths[chunk_start:chunk_end]
            chunk_audios = audio_paths[chunk_start:chunk_end]
            chunk_size = len(chunk_refs)
            
            logger.info(f"Processing chunk {chunk_start//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size} ({chunk_size} samples)")
            
            # Preprocess batch
            preprocess_start = time.time()
            data = self.data_processor.preprocess_batch(
                chunk_refs, chunk_audios, no_crop=no_crop, num_workers=4
            )
            timings['preprocess'] += time.time() - preprocess_start
            
            # Inference
            use_amp = hasattr(self, 'use_amp') and self.use_amp
            use_amp_fp16 = hasattr(self, 'use_amp_fp16') and self.use_amp_fp16
            
            inference_start = time.time()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            if use_amp or use_amp_fp16:
                with torch.cuda.amp.autocast(dtype=torch.float16 if use_amp_fp16 else torch.float32):
                    d_hat = self.G_opt.inference_batch(
                        data=data,
                        a_cfg_scale=a_cfg_scale,
                        r_cfg_scale=r_cfg_scale,
                        e_cfg_scale=e_cfg_scale,
                        emo=emo,
                        nfe=nfe,
                        seed=seed,
                        batch_size=4  # Inner batch size for memory efficiency
                    )['d_hat']
            else:
                d_hat = self.G_opt.inference_batch(
                    data=data,
                    a_cfg_scale=a_cfg_scale,
                    r_cfg_scale=r_cfg_scale,
                    e_cfg_scale=e_cfg_scale,
                    emo=emo,
                    nfe=nfe,
                    seed=seed,
                    batch_size=4
                )['d_hat']
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            timings['inference'] += time.time() - inference_start
            
            # Save videos
            save_start = time.time()
            chunk_paths = self.save_video(d_hat.float(), res_video_path, None)
            timings['save'] += time.time() - save_start
            
            all_results.extend(chunk_paths)
        
        timings['total'] = time.time() - total_start
        
        if verbose:
            self._print_batch_timings(timings, total_samples)
        
        return all_results
    
    def _run_benchmark(self, data, a_cfg_scale, r_cfg_scale, e_cfg_scale, emo, nfe, seed):
        """Run benchmarking"""
        warmup_runs = 3
        benchmark_runs = 10
        
        print("Warming up...")
        for _ in range(warmup_runs):
            _ = self.G_opt.inference(
                data=data,
                a_cfg_scale=a_cfg_scale,
                r_cfg_scale=r_cfg_scale,
                e_cfg_scale=e_cfg_scale,
                emo=emo,
                nfe=nfe,
                seed=seed
            )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        print(f"Running {benchmark_runs} benchmark iterations...")
        start_time = time.time()
        for _ in range(benchmark_runs):
            _ = self.G_opt.inference(
                data=data,
                a_cfg_scale=a_cfg_scale,
                r_cfg_scale=r_cfg_scale,
                e_cfg_scale=e_cfg_scale,
                emo=emo,
                nfe=nfe,import os, torch, random, cv2, torchvision, librosa, datetime, tempfile, face_alignment
import numpy as np
import albumentations as A
import albumentations.pytorch.transforms as A_pytorch
from tqdm import tqdm
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor
from models.float.FLOAT import FLOAT
from options.base_options import BaseOptions
import time
import uuid
import logging
import struct
import wave
import subprocess
import torch
import numpy as np
import time
import shutil
import os
logger = logging.getLogger(__name__)

def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Setup a logger with console output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger(__name__, level=logging.INFO)


class SimpleVideoEncoder:
    """
    A from-scratch video encoder that creates AVI files with Motion JPEG encoding.
    This is a simple, pure-Python implementation that doesn't rely on ffmpeg.
    """
    
    def __init__(self, width: int, height: int, fps: int, output_path: str):
        self.width = width
        self.height = height
        self.fps = fps
        self.output_path = output_path
        self.frame_count = 0
        self.frames = []
        
    def add_frame(self, frame: np.ndarray):
        """Add a frame to the video. Frame should be RGB uint8 numpy array (H, W, 3)"""
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Frame shape {frame.shape} doesn't match video size ({self.height}, {self.width}, 3)")
        
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.frames.append(bgr_frame)
        self.frame_count += 1
    
    def write_avi_with_audio(self, audio_path: str = None):
        """Write frames to AVI file using OpenCV VideoWriter"""
        if self.frame_count == 0:
            raise ValueError("No frames to write!")
        
        # Use MJPEG codec (widely supported, no external dependencies)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        
        # Create temporary video file without audio
        temp_video = self.output_path.replace('.mp4', '_temp.avi').replace('.avi', '_temp.avi')
        
        out = cv2.VideoWriter(temp_video, fourcc, self.fps, (self.width, self.height))
        
        if not out.isOpened():
            raise RuntimeError("Failed to open video writer")
        
        logger.info(f"Writing {self.frame_count} frames to {temp_video}...")
        for frame in tqdm(self.frames, desc="Encoding video"):
            out.write(frame)
        
        out.release()
        logger.info(f"✅ Video encoding complete: {temp_video}")
        
        # If audio is provided, combine using OpenCV (limited) or just save video-only
        if audio_path and os.path.exists(audio_path):
            logger.warning("Audio merging without ffmpeg is limited. Saving video-only version.")
            logger.info(f"To add audio manually, use: ffmpeg -i {temp_video} -i {audio_path} -c:v copy -c:a aac {self.output_path}")
        
        # Rename temp to final output
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        os.rename(temp_video, self.output_path)
        
        return self.output_path
    
    def write(self):
        """Write frames without audio"""
        return self.write_avi_with_audio(audio_path=None)


class DataProcessor:
    def __init__(self, opt):
        self.opt = opt
        self.fps = opt.fps
        self.sampling_rate = opt.sampling_rate
        self.input_size = opt.input_size
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
        
        if os.path.exists(opt.wav2vec_model_path):
            self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(opt.wav2vec_model_path, local_files_only=True)
        else:
            self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(opt.wav2vec_model_path)
        
        self.transform = A.Compose([
            A.Resize(height=opt.input_size, width=opt.input_size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
            A_pytorch.ToTensorV2(),
        ])
    
    @torch.no_grad()
    def process_img(self, img:np.ndarray) -> np.ndarray:
        mult = 360. / img.shape[0]
        resized_img = cv2.resize(img, dsize=(0, 0), fx = mult, fy = mult, 
                                interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
        bboxes = self.fa.face_detector.detect_from_image(resized_img)
        bboxes = [(int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), score) 
                  for (x1, y1, x2, y2, score) in bboxes if score > 0.95]
        bboxes = bboxes[0]
        
        bsy = int((bboxes[3] - bboxes[1]) / 2)
        bsx = int((bboxes[2] - bboxes[0]) / 2)
        my = int((bboxes[1] + bboxes[3]) / 2)
        mx = int((bboxes[0] + bboxes[2]) / 2)
        bs = int(max(bsy, bsx) * 1.6)
        
        img = cv2.copyMakeBorder(img, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=0)
        my, mx = my + bs, mx + bs
        crop_img = img[my - bs:my + bs, mx - bs:mx + bs]
        crop_img = cv2.resize(crop_img, dsize = (self.input_size, self.input_size), 
                             interpolation = cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
        return crop_img
    
    def default_img_loader(self, path) -> np.ndarray:
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def default_aud_loader(self, path: str) -> torch.Tensor:
        speech_array, sampling_rate = librosa.load(path, sr = self.sampling_rate)
        return self.wav2vec_preprocessor(speech_array, sampling_rate = sampling_rate, 
                                        return_tensors = 'pt').input_values[0]
    
    def preprocess(self, ref_paths:list, audio_paths:list, no_crop:bool) -> dict:
        s_list = []
        a_list = []
        for ref_path, audio_path in zip(ref_paths, audio_paths):
            s = self.default_img_loader(ref_path)
            if not no_crop:
                s = self.process_img(s)
            s = self.transform(image=s)['image']
            a = self.default_aud_loader(audio_path)
            s_list.append(s)
            a_list.append(a)
        
        s_batch = torch.stack(s_list, dim=0)
        a_batch = torch.stack(a_list, dim=0)
        
        return {'s': s_batch, 'a': a_batch, 'p': None, 'e': None}


class OptimizedInferenceAgent:
    def __init__(self, opt):
        torch.cuda.empty_cache()
        self.opt = opt
        self.rank = opt.rank
        self.optimization_mode = opt.optimization_mode if hasattr(opt, 'optimization_mode') else 'torch_compile'
        self.use_fp16 = opt.use_fp16 if hasattr(opt, 'use_fp16') else False
        
        # Load Model
        self.load_model()
        self.load_weight(opt.ckpt_path, rank=self.rank)
        self.G.to(self.rank)
        self.G.eval()
        
        # Apply optimization
        self.apply_optimization()
        
        # Load Data Processor
        self.data_processor = DataProcessor(opt)
        
        print(f"✓ Model loaded with {self.optimization_mode} optimization")
    
    def load_model(self) -> None:
        self.G = FLOAT(self.opt)
    
    def load_weight(self, checkpoint_path: str, rank: int) -> None:
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        with torch.no_grad():
            for model_name, model_param in self.G.named_parameters():
                if model_name in state_dict:
                    model_param.copy_(state_dict[model_name].to(rank))
                elif "wav2vec2" in model_name:
                    pass
                else:
                    print(f"! Warning; {model_name} not found in state_dict.")
        del state_dict
    
    def apply_optimization(self):
        """Apply various optimization strategies"""
        
        if self.optimization_mode == 'torch_compile':
            try:
                if hasattr(torch, 'compile'):
                    print("Applying torch.compile optimization...")
                    self.G_opt = torch.compile(
                        self.G,
                        mode='max-autotune',
                        fullgraph=False
                    )
                    print("✓ torch.compile applied successfully")
                else:
                    print("! torch.compile not available (requires PyTorch 2.0+)")
                    self.G_opt = self.G
            except Exception as e:
                print(f"! torch.compile failed: {e}")
                self.G_opt = self.G
        
        elif self.optimization_mode == 'channels_last':
            print("Applying channels_last memory format...")
            try:
                for module in self.G.modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
                        module = module.to(memory_format=torch.channels_last)
                self.G_opt = self.G
                print("✓ Channels last format applied")
            except Exception as e:
                print(f"! Channels last conversion failed: {e}, skipping...")
                self.G_opt = self.G
        
        elif self.optimization_mode == 'amp':
            print("Using Automatic Mixed Precision (AMP)...")
            self.G_opt = self.G
            self.use_amp = True
            print("✓ AMP enabled")
        
        elif self.optimization_mode == 'cudnn_benchmark':
            print("Enabling cuDNN benchmark mode...")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            self.G_opt = self.G
            print("✓ cuDNN benchmark enabled")
        
        elif self.optimization_mode == 'all':
            print("Applying all optimizations...")
            
            try:
                for module in self.G.modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
                        module = module.to(memory_format=torch.channels_last)
                print("✓ Channels last applied to conv layers")
            except Exception as e:
                print(f"! Channels last skipped: {e}")
            
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            print("✓ cuDNN benchmark enabled")
            
            if hasattr(torch, 'compile'):
                try:
                    self.G_opt = torch.compile(
                        self.G,
                        mode='max-autotune',
                        fullgraph=False
                    )
                    print("✓ All optimizations applied")
                except Exception as e:
                    print(f"! torch.compile failed: {e}")
                    self.G_opt = self.G
                    print("✓ Partial optimizations applied")
            else:
                self.G_opt = self.G
                print("✓ Partial optimizations applied")
        
        else:
            print("No optimization applied")
            self.G_opt = self.G
        
        if self.use_fp16:
            print("Enabling FP16 inference...")
            self.use_amp_fp16 = True
            print("✓ FP16 will be applied via autocast during inference")
    
# Replace the save_video method in inference_optimized.py

    def save_video(self, vid_target_recon: "torch.Tensor", video_path: str, audio_path: str) -> str:
        """
        Save video with proper CUDA synchronization to prevent corruption.
        
        CRITICAL FIXES:
        1. Explicit GPU synchronization before tensor operations
        2. Ensure tensor is fully on CPU before FFmpeg access
        3. Add delay for NVENC encoder initialization
        """
        start_time = time.time()

        # Sanity: ffmpeg present
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found in PATH")


        # Step 1: Prepare tensor -> CPU uint8 numpy efficiently
        t0 = time.time()
        vid_t = vid_target_recon.detach()

        # ===== CRITICAL FIX 2: Blocking CPU transfer =====
        if vid_t.is_cuda:
            # Use blocking transfer to ensure tensor is fully on CPU
            vid_t = vid_t.to("cpu", non_blocking=False)  # Changed to blocking!


        # Expect shape (B, C, H, W)
        if vid_t.ndim != 4:
            raise ValueError("vid_target_recon must be a 4D tensor (T, C, H, W)")

        # Permute to (T, H, W, C) and ensure contiguous
        vid_t = vid_t.permute(0, 2, 3, 1).contiguous()

        # Convert float in [-1,1] => uint8 [0,255]
        if torch.is_floating_point(vid_t):
            vid_t = (vid_t.clamp(-1.0, 1.0) * 127.5 + 127.5).to(torch.uint8)
        else:
            vid_t = vid_t.to(torch.uint8)

        # Convert to numpy (this should be instant now since tensor is on CPU)
        vid_np = vid_t.numpy()
        T, H, W, C = vid_np.shape
        
        # ===== CRITICAL FIX 3: Verify numpy array is valid =====
        if not vid_np.flags['C_CONTIGUOUS']:
            vid_np = np.ascontiguousarray(vid_np)
        
        if C not in (3, 4):
            raise ValueError(f"Expected 3 or 4 channels, got {C}")

        fps = int(getattr(self.opt, "fps", 25))
        tensor_prep_time = time.time() - t0
        logger.info(f"⏱️ Tensor -> numpy: {tensor_prep_time:.3f}s (T={T}, H={H}, W={W}, C={C})")

        # Build ffmpeg command
        pix_fmt_in = "rgba" if C == 4 else "rgb24"
        size_str = f"{W}x{H}"

        base_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", pix_fmt_in,
            "-s", size_str,
            "-r", str(fps),
            "-i", "-",  # read video from stdin
        ]

        #if audio_path is not None:
        #    base_cmd += ["-i", audio_path]

        # ===== CRITICAL FIX 4: More conservative NVENC settings =====
        # Use settings that are more stable for first-time initialization
        vcodec = "h264_nvenc"
        encode_opts = [
            "-c:v", vcodec,
            "-preset", "p1",           # Fastest preset
            "-tune", "hq",
            "-rc", "constqp",          # Constant QP for stability
            "-qp", "23",
            "-delay", "0",             # Minimize encoder latency
            "-no-scenecut", "1",       # Disable scene detection for consistency
            "-forced-idr", "1",        # Force IDR frames
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]

        #if audio_path is not None:
        #    encode_opts += ["-c:a", "aac", "-b:a", "192k", "-shortest"]

        cmd = base_cmd + encode_opts + [video_path]

        # ===== CRITICAL FIX 5: Better error handling and retry logic =====
        encode_start = time.time()
        proc = None
        max_retries = 2
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    bufsize=65536  # 64KB buffer for smoother writes
                )

                # Write frames with small delays for first-time stability
                write_start = time.time()
                chunk_size = 25  # Write frames in small chunks
                
                for chunk_start in range(0, T, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, T)
                    
                    for i in range(chunk_start, chunk_end):
                        frame = vid_np[i]
                        frame_bytes = frame.tobytes()
                        
                        # Write with error checking
                        try:
                            proc.stdin.write(frame_bytes)
                        except BrokenPipeError:
                            logger.error("FFmpeg pipe broken during write")
                            raise
                    
                    # Small delay between chunks on first request (helps NVENC stabilize)
                    if retry_count == 0 and chunk_start == 0:
                        time.sleep(0.01)
                
                proc.stdin.close()

                # Wait with timeout
                ret = proc.wait(timeout=300)
                
                if ret != 0:
                    stderr = proc.stderr.read().decode(errors="ignore")
                    
                    # Check if this is an NVENC initialization error
                    if "CUDA_ERROR" in stderr or "initialization error" in stderr:
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.warning(f"⚠️ NVENC init error, retry {retry_count}/{max_retries}")
                            time.sleep(0.5)  # Wait before retry
                            # Force CUDA sync again
                            if torch.cuda.is_available():
                                1
                            continue
                    
                    raise RuntimeError(f"ffmpeg exited with code {ret}: {stderr}")

                encode_time = time.time() - encode_start
                logger.info(f"✅ Direct FFmpeg encode: {encode_time:.3f}s")
                break  # Success!

            except subprocess.TimeoutExpired:
                if proc:
                    proc.kill()
                raise RuntimeError("FFmpeg encoding timeout")
                
            except Exception as e:
                if proc:
                    try:
                        proc.kill()
                        stderr = proc.stderr.read().decode(errors="ignore")
                        logger.error(f"FFmpeg stderr:\n{stderr[:500]}")
                    except:
                        pass
                
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"❌ Encoding failed after {max_retries} retries: {e}")
                    raise
                
                logger.warning(f"⚠️ Encoding attempt {retry_count} failed, retrying...")
                time.sleep(0.5)

        total_time = time.time() - start_time
        logger.info(f"✅ Total save: {total_time:.3f}s -> {video_path}")
        return video_path
    
    @torch.no_grad()
    def run_inference(
        self,
        res_video_paths: list,
        ref_paths: list,
        audio_paths: list,
        a_cfg_scale: float = 2.0,
        r_cfg_scale: float = 1.0,
        e_cfg_scale: float = 1.0,
        emo: str = 'S2E',
        nfe: int = 10,
        no_crop: bool = False,
        seed: int = 25,
        verbose: bool = True,
        benchmark: bool = False
    ) -> list:
        
        timings = {}
        total_start = time.time()
        
        # Preprocess data
        preprocess_start = time.time()
        data = self.data_processor.preprocess(ref_paths, audio_paths, no_crop=no_crop)
        timings['preprocess'] = time.time() - preprocess_start
        
        if verbose:
            print(f"> [Done] Preprocess ({timings['preprocess']:.3f}s)")
        
        # Benchmark mode
        if benchmark:
            warmup_runs = 3
            benchmark_runs = 10
            
            print("Warming up...")
            for _ in range(warmup_runs):
                _ = self.G_opt.inference(
                    data=data,
                    a_cfg_scale=a_cfg_scale,
                    r_cfg_scale=r_cfg_scale,
                    e_cfg_scale=e_cfg_scale,
                    emo=emo,
                    nfe=nfe,
                    seed=seed
                )
            
            
            
            print(f"Running {benchmark_runs} benchmark iterations...")
            start_time = time.time()
            for _ in range(benchmark_runs):
                _ = self.G_opt.inference(
                    data=data,
                    a_cfg_scale=a_cfg_scale,
                    r_cfg_scale=r_cfg_scale,
                    e_cfg_scale=e_cfg_scale,
                    emo=emo,
                    nfe=nfe,
                    seed=seed
                )
            
            end_time = time.time()
            
            avg_time = (end_time - start_time) / benchmark_runs
            print(f"Average inference time: {avg_time*1000:.2f} ms")
            print(f"FPS: {1.0/avg_time:.2f}")
        
        
        # Move data to GPU
        data = {k: v.to(self.rank) for k, v in data.items() if isinstance(v, torch.Tensor)}
        
        # Actual inference
        use_amp = hasattr(self, 'use_amp') and self.use_amp
        use_amp_fp16 = hasattr(self, 'use_amp_fp16') and self.use_amp_fp16
        
        inference_start = time.time()
        1 if torch.cuda.is_available() else None
        
        if use_amp or use_amp_fp16:
            with torch.cuda.amp.autocast(dtype=torch.float16 if use_amp_fp16 else torch.float32):
                d_hat = self.G_opt.inference(
                    data=data,
                    a_cfg_scale=a_cfg_scale,
                    r_cfg_scale=r_cfg_scale,
                    e_cfg_scale=e_cfg_scale,
                    emo=emo,
                    nfe=nfe,
                    seed=seed
                )['d_hat']
        else:
            d_hat = self.G_opt.inference(
                data=data,
                a_cfg_scale=a_cfg_scale,
                r_cfg_scale=r_cfg_scale,
                e_cfg_scale=e_cfg_scale,
                emo=emo,
                nfe=nfe,
                seed=seed
            )['d_hat']
        
        1 if torch.cuda.is_available() else None
        timings['inference'] = time.time() - inference_start
        
        if verbose:
            print(f"> [Done] Inference ({timings['inference']:.3f}s)")
        
        # Save result
        save_start = time.time()
        for i in range(d_hat.shape[0]):
            self.save_video(d_hat[i].unsqueeze(0).float(), res_video_paths[i], audio_paths[i])
        timings['save'] = time.time() - save_start
        
        timings['total'] = time.time() - total_start
        
        if verbose:
            print(f"> [Done] Save videos ({timings['save']:.3f}s)")
            print(f"\n{'='*50}")
            print(f"⏱️  PERFORMANCE SUMMARY")
            print(f"{'='*50}")
            print(f"  Preprocess:  {timings['preprocess']:>8.3f}s")
            print(f"  Inference:   {timings['inference']:>8.3f}s")
            print(f"  Save Video:  {timings['save']:>8.3f}s")
            print(f"  {'─'*40}")
            print(f"  Total Time:  {timings['total']:>8.3f}s")
            print(f"  FPS:         {d_hat.shape[0]/timings['inference']:>8.2f}")
            print(f"{'='*50}")
            for path in res_video_paths:
                print(f"> Result saved at {path}")
        
        return res_video_paths


class InferenceOptions(BaseOptions):
    def __init__(self):
        super().__init__()
    
    def initialize(self, parser):
        super().initialize(parser)
        parser.add_argument("--ref_path", default=None, nargs='+', help='reference image(s)')
        parser.add_argument('--aud_path', default=None, nargs='+', help='audio file(s)')
        parser.add_argument('--emo', default=None, type=str, help='emotion', 
                          choices=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
        parser.add_argument('--no_crop', action='store_true', help='skip face cropping')
        parser.add_argument('--res_video_path', default=None, nargs='+', help='output video path(s)')
        parser.add_argument('--ckpt_path', default="./checkpoints/float.pth", type=str, help='checkpoint path')
        parser.add_argument('--res_dir', default="./results", type=str, help='result directory')
        
        parser.add_argument('--optimization_mode', default='all', type=str, 
                          choices=['none', 'torch_compile', 'channels_last', 'amp', 'cudnn_benchmark', 'all'],
                          help='Optimization strategy to use')
        parser.add_argument('--use_fp16', action='store_true', help='Use FP16 precision')
        parser.add_argument('--benchmark', action='store_true', help='Run benchmark mode')
        
        return parser


if __name__ == '__main__':
    opt = InferenceOptions().parse()
    opt.rank, opt.ngpus = 0, 1
    
    if not opt.ref_path or not opt.aud_path:
        raise ValueError("--ref_path and --aud_path are required.")
    
    print(f"\n{'='*60}")
    print(f"FLOAT Optimized Inference (No FFmpeg)")
    print(f"Batch size: {len(opt.ref_path)}")
    print(f"Optimization: {opt.optimization_mode}")
    print(f"FP16: {opt.use_fp16}")
    print(f"{'='*60}\n")
    
    agent = OptimizedInferenceAgent(opt)
    
    os.makedirs(opt.res_dir, exist_ok=True)
    
    ref_paths = opt.ref_path
    aud_paths = opt.aud_path
    
    if len(ref_paths) != len(aud_paths):
        raise ValueError("The number of reference images and audio files must be the same.")

    if opt.res_video_path and len(ref_paths) != len(opt.res_video_path):
        raise ValueError("The number of output video paths must match the number of inputs.")

    if opt.res_video_path:
        if len(opt.res_video_path) != len(ref_paths):
            raise ValueError("The number of output video paths must match the number of inputs.")
        res_video_paths = opt.res_video_path
    else:
        res_video_paths = []
        call_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        opt_suffix = f"{opt.optimization_mode}"
        if opt.use_fp16:
            opt_suffix += "-fp16"
            
        for i, (ref_path, aud_path) in enumerate(zip(ref_paths, aud_paths)):
            video_name = os.path.splitext(os.path.basename(ref_path))[0]
            audio_name = os.path.splitext(os.path.basename(aud_path))[0]
            res_video_paths.append(
                os.path.join(opt.res_dir, "%s-%s-%s-nfe%s-seed%s-acfg%s-ecfg%s-%s-%s.avi" \
                    % (call_time, video_name, audio_name, opt.nfe, opt.seed, opt.a_cfg_scale, 
                       opt.e_cfg_scale, opt.emo, opt_suffix))
            )
    
    agent.run_inference(
        res_video_paths,
        ref_paths,
        aud_paths,
        a_cfg_scale=opt.a_cfg_scale,
        r_cfg_scale=opt.r_cfg_scale,
        e_cfg_scale=opt.e_cfg_scale,
        emo=opt.emo,
        nfe=opt.nfe,
        no_crop=opt.no_crop,
        seed=opt.seed,
        verbose=True,
        benchmark=opt.benchmark
    )
                seed=seed
            )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / benchmark_runs
        print(f"Average inference time: {avg_time*1000:.2f} ms")
        print(f"FPS: {1.0/avg_time:.2f}")
    
    def _print_timings(self, timings):
        """Print timing summary"""
        print(f"\n{'='*50}")
        print(f"⏱️  PERFORMANCE SUMMARY")
        print(f"{'='*50}")
        print(f"  Preprocess:  {timings['preprocess']:>8.3f}s")
        print(f"  Inference:   {timings['inference']:>8.3f}s")
        print(f"  Save Video:  {timings['save']:>8.3f}s")
        print(f"  {'─'*40}")
        print(f"  Total Time:  {timings['total']:>8.3f}s")
        print(f"  FPS:         {1.0/timings['inference']:>8.2f}")
        print(f"{'='*50}")
    
    def _print_batch_timings(self, timings, total_samples):
        """Print batch timing summary"""
        print(f"\n{'='*50}")
        print(f"⏱️  BATCH PERFORMANCE SUMMARY ({total_samples} samples)")
        print(f"{'='*50}")
        print(f"  Preprocess:  {timings['preprocess']:>8.3f}s ({timings['preprocess']/total_samples:.3f}s/sample)")
        print(f"  Inference:   {timings['inference']:>8.3f}s ({timings['inference']/total_samples:.3f}s/sample)")
        print(f"  Save Video:  {timings['save']:>8.3f}s ({timings['save']/total_samples:.3f}s/sample)")
        print(f"  {'─'*40}")
        print(f"  Total Time:  {timings['total']:>8.3f}s ({timings['total']/total_samples:.3f}s/sample)")
        print(f"  Throughput:  {total_samples/timings['total']:>8.2f} samples/s")
        print(f"{'='*50}")


class InferenceOptions(BaseOptions):
    def __init__(self):
        super().__init__()
    
    def initialize(self, parser):
        super().initialize(parser)
        parser.add_argument("--ref_path", default=None, type=str, nargs='+', help='reference image(s)')
        parser.add_argument('--aud_path', default=None, type=str, nargs='+', help='audio file(s)')
        parser.add_argument('--emo', default=None, type=str, help='emotion', 
                          choices=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
        parser.add_argument('--no_crop', action='store_true', help='skip face cropping')
        parser.add_argument('--res_video_path', default=None, type=str, help='output video path or directory')
        parser.add_argument('--ckpt_path', default="./checkpoints/float.pth", type=str, help='checkpoint path')
        parser.add_argument('--res_dir', default="./results", type=str, help='result directory')
        
        parser.add_argument('--optimization_mode', default='all', type=str, 
                          choices=['none', 'torch_compile', 'channels_last', 'amp', 'cudnn_benchmark', 'all'],
                          help='Optimization strategy to use')
        parser.add_argument('--use_fp16', action='store_true', help='Use FP16 precision')
        parser.add_argument('--benchmark', action='store_true', help='Run benchmark mode')
        parser.add_argument('--batch_size', default=None, type=int, help='Batch size for batch processing')
        
        return parser


if __name__ == '__main__':
    opt = InferenceOptions().parse()
    opt.rank, opt.ngpus = 0, 1
    
    print(f"\n{'='*60}")
    print(f"FLOAT Optimized Inference (Batch Support)")
    print(f"Optimization: {opt.optimization_mode}")
    print(f"FP16: {opt.use_fp16}")
    print(f"{'='*60}\n")
    
    agent = OptimizedInferenceAgent(opt)
    
    os.makedirs(opt.res_dir, exist_ok=True)
    
    # Handle single or batch input
    ref_path = opt.ref_path if isinstance(opt.ref_path, list) else [opt.ref_path] if opt.ref_path else None
    aud_path = opt.aud_path if isinstance(opt.aud_path, list) else [opt.aud_path] if opt.aud_path else None
    
    if ref_path is None or aud_path is None:
        raise ValueError("Must provide --ref_path and --aud_path")
    
    # Single vs batch
    is_batch = len(ref_path) > 1
    
    if opt.res_video_path is None:
        if is_batch:
            # Create directory for batch output
            call_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            opt_suffix = f"{opt.optimization_mode}"
            if opt.use_fp16:
                opt_suffix += "-fp16"
            res_video_path = os.path.join(opt.res_dir, f"batch_{call_time}_{opt_suffix}")
        else:
            # Single file output
            video_name = os.path.splitext(os.path.basename(ref_path[0]))[0]
            audio_name = os.path.splitext(os.path.basename(aud_path[0]))[0]
            call_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            opt_suffix = f"{opt.optimization_mode}"
            if opt.use_fp16:
                opt_suffix += "-fp16"
            res_video_path = os.path.join(opt.res_dir, "%s-%s-%s-nfe%s-seed%s-acfg%s-ecfg%s-%s-%s.mp4" \
                % (call_time, video_name, audio_name, opt.nfe, opt.seed, opt.a_cfg_scale, 
                   opt.e_cfg_scale, opt.emo, opt_suffix))
    else:
        res_video_path = opt.res_video_path
    
    # Run inference
    if is_batch and len(ref_path) == len(aud_path):
        result_paths = agent.run_inference(
            res_video_path,
            ref_path,
            aud_path,
            a_cfg_scale=opt.a_cfg_scale,
            r_cfg_scale=opt.r_cfg_scale,
            e_cfg_scale=opt.e_cfg_scale,
            emo=opt.emo,
            nfe=opt.nfe,
            no_crop=opt.no_crop,
            seed=opt.seed,
            verbose=True,
            batch_size=opt.batch_size
        )
        print(f"\n✅ Batch processing complete! {len(result_paths)} videos saved to {res_video_path}")
    else:
        result_path = agent.run_inference(
            res_video_path,
            ref_path[0],
            aud_path[0],
            a_cfg_scale=opt.a_cfg_scale,
            r_cfg_scale=opt.r_cfg_scale,
            e_cfg_scale=opt.e_cfg_scale,
            emo=opt.emo,
            nfe=opt.nfe,
            no_crop=opt.no_crop,
            seed=opt.seed,
            verbose=True,
            benchmark=opt.benchmark
        )
        print(f"\n✅ Single inference complete! Video saved to {result_path}")