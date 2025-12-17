import os
import torch, math
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
import torch.nn as nn
import torch.nn.functional as F
import asyncio
from torchdiffeq import odeint
from transformers import Wav2Vec2Config
from transformers.modeling_outputs import BaseModelOutput
import torch
import subprocess
import time
import shutil
import numpy as np
from typing import Generator
import traceback
import threading
import queue
from dataclasses import dataclass
    
import torch
import subprocess
import time
import shutil
import numpy as np
from typing import Generator
import traceback
import threading
import queue
from dataclasses import dataclass
@dataclass
class FrameData:
    """Container for frame data passed between producer and consumer"""
    frame_np: np.ndarray
    frame_idx: int
    is_last: bool = False

from models.wav2vec2 import Wav2VecModel
from models.wav2vec2_ser import Wav2Vec2ForSpeechClassification

from models import BaseModel
from models.float.generator import Generator
from models.float.FMT import FlowMatchingTransformer
import traceback
torch.backends.cudnn.benchmark = True
######## Main Phase 2 model ########	
class StreamingVideoSaver:

    @torch.no_grad()
    def inference_and_save(
        self, 
        data: dict, 
        video_path: str,
        audio_path: str = None,
        a_cfg_scale=None, 
        r_cfg_scale=None,
        e_cfg_scale=None, 
        emo=None, 
        nfe=10, 
        seed=None
    ) -> str:
        """
        Generate and save video using producer-consumer pattern.
        Producer: Decodes frames and adds to queue
        Consumer: Takes frames from queue and writes to FFmpeg
        Returns path to saved video.
        """
        start_time = time.time()
        
        # Sanity: ffmpeg present
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found in PATH")
        
        # Setup encoding parameters
        s, a = data['s'], data['a']
        s_r, r_s_lambda, s_r_feats = self.encode_image_into_latent(s.to(self.opt.rank))
        
        if 's_r' in data:
            r_s = self.encode_identity_into_motion(s_r)
        else:
            r_s = self.motion_autoencoder.dec.direction(r_s_lambda)
        data['r_s'] = r_s
        
        if a_cfg_scale is None: a_cfg_scale = self.opt.a_cfg_scale
        if r_cfg_scale is None: r_cfg_scale = self.opt.r_cfg_scale
        if e_cfg_scale is None: e_cfg_scale = self.opt.e_cfg_scale
        
        fps = int(getattr(self.opt, "fps", 25))
        
        # Create queue for producer-consumer pattern
        # Size = 10 means buffer up to 10 frames aheadStreamingVideoSaver._start_ffmpeg_process
        frame_queue = queue.Queue(maxsize=10)
        
        # Shared state
        producer_error = {'error': None}
        consumer_error = {'error': None}
        stats = {'frames_produced': 0, 'frames_consumed': 0}
        
        # Producer thread: Generate and decode frames
        def producer():
            """Decode frames and put them in the queue"""
            try:
                print("[PRODUCER] Starting frame generation...")
                
                # Generator that yields decoded frames
                frame_generator = self._generate_and_decode_frames(
                    data=data,
                    s_r=s_r,
                    s_r_feats=s_r_feats,
                    a_cfg_scale=a_cfg_scale,
                    r_cfg_scale=r_cfg_scale,
                    e_cfg_scale=e_cfg_scale,
                    emo=emo,
                    nfe=nfe,
                    seed=seed
                )
                
                for frame_idx, frame_tensor in enumerate(frame_generator):
                    # Check if consumer hit an error
                    if consumer_error['error'] is not None:
                        print(f"[PRODUCER] Consumer error detected, stopping production")
                        break
                    
                    # Convert frame to numpy
                    frame_np = self._tensor_to_numpy(frame_tensor)
                    
                    # Put frame in queue (blocks if queue is full)
                    frame_data = FrameData(
                        frame_np=frame_np,
                        frame_idx=frame_idx,
                        is_last=False
                    )
                    frame_queue.put(frame_data)
                    stats['frames_produced'] += 1
                    
                    if frame_idx % 10 == 0:
                        print(f"[PRODUCER] Generated frame {frame_idx}, queue size: {frame_queue.qsize()}")
                
                # Send sentinel to signal completion
                frame_queue.put(FrameData(frame_np=None, frame_idx=-1, is_last=True))
                print(f"[PRODUCER] ✅ Finished: {stats['frames_produced']} frames")
                
            except Exception as e:
                producer_error['error'] = e
                print(f"[PRODUCER] ❌ Error: {e}\n{traceback.format_exc()}")
                # Put sentinel even on error to unblock consumer
                try:
                    frame_queue.put(FrameData(frame_np=None, frame_idx=-1, is_last=True))
                except:
                    pass
        
        # Consumer thread: Write frames to FFmpeg
        def consumer():
            """Take frames from queue and write to FFmpeg"""
            ffmpeg_proc = None
            try:
                print("[CONSUMER] Waiting for frames...")
                
                # Sync GPU before starting
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    time.sleep(0.05)
                
                while True:
                    # Get frame from queue (blocks if queue is empty)
                    frame_data = frame_queue.get()
                    
                    # Check for sentinel (end of stream)
                    if frame_data.is_last:
                        print(f"[CONSUMER] Received end signal")
                        break
                    
                    # Check if producer hit an error
                    if producer_error['error'] is not None:
                        print(f"[CONSUMER] Producer error detected, stopping consumption")
                        break
                    
                    frame_np = frame_data.frame_np
                    frame_idx = frame_data.frame_idx
                    
                    # Initialize FFmpeg on first frame
                    if ffmpeg_proc is None:
                        # Handle batch dimension if present
                        if frame_np.ndim == 4:  # [1, H, W, C]
                            H, W, C = frame_np.shape[1], frame_np.shape[2], frame_np.shape[3]
                        else:  # [H, W, C]
                            H, W, C = frame_np.shape[0], frame_np.shape[1], frame_np.shape[2]
                        
                        ffmpeg_proc = self._start_ffmpeg_process(video_path, H, W, C, fps)
                        print(f"[CONSUMER] Started FFmpeg: {W}x{H}x{C} @ {fps}fps")
                    
                    # Write frame to FFmpeg
                    try:
                        # Handle batch dimension if present
                        if frame_np.ndim == 4:
                            frame_np = frame_np[0]
                        
                        frame_bytes = frame_np.tobytes()
                        ffmpeg_proc.stdin.write(frame_bytes)
                        stats['frames_consumed'] += 1
                        
                        if frame_idx % 10 == 0:
                            print(f"[CONSUMER] Wrote frame {frame_idx}, queue size: {frame_queue.qsize()}")
                        
                        # Mark task as done
                        frame_queue.task_done()
                        
                    except BrokenPipeError:
                        stderr = ffmpeg_proc.stderr.read().decode(errors="ignore")
                        raise RuntimeError(f"FFmpeg pipe broken at frame {frame_idx}: {stderr}")
                
                # Close FFmpeg and wait for completion
                if ffmpeg_proc is not None:
                    print(f"[CONSUMER] Closing FFmpeg stdin after {stats['frames_consumed']} frames")
                    ffmpeg_proc.stdin.close()
                    
                    ret = ffmpeg_proc.wait(timeout=300)
                    if ret != 0:
                        stderr = ffmpeg_proc.stderr.read().decode(errors="ignore")
                        raise RuntimeError(f"FFmpeg exited with code {ret}: {stderr}")
                    
                    print(f"[CONSUMER] ✅ FFmpeg encoding complete")
                
            except Exception as e:
                consumer_error['error'] = e
                print(f"[CONSUMER] ❌ Error: {e}\n{traceback.format_exc()}")
                if ffmpeg_proc:
                    try:
                        ffmpeg_proc.kill()
                        stderr = ffmpeg_proc.stderr.read().decode(errors="ignore")
                        print(f"FFmpeg stderr:\n{stderr[:500]}")
                    except:
                        pass
        
        # Start producer and consumer threads
        producer_thread = threading.Thread(target=producer, name="FrameProducer")
        consumer_thread = threading.Thread(target=consumer, name="FrameConsumer")
        
        print("[STREAMING] Starting producer-consumer threads...")
        producer_thread.start()
        consumer_thread.start()
        
        # Wait for both threads to complete
        producer_thread.join()
        consumer_thread.join()
        
        # Check for errors
        if producer_error['error'] is not None:
            raise RuntimeError(f"Producer error: {producer_error['error']}")
        if consumer_error['error'] is not None:
            raise RuntimeError(f"Consumer error: {consumer_error['error']}")
        
        total_time = time.time() - start_time
        print(f"\n[STREAMING] ✅ Complete:")
        print(f"  - Frames produced: {stats['frames_produced']}")
        print(f"  - Frames consumed: {stats['frames_consumed']}")
        print(f"  - Total time: {total_time:.3f}s")
        print(f"  - Output: {video_path}")
        
        return video_path
    
    def _generate_and_decode_frames(
        self, 
        data: dict,
        s_r: torch.Tensor,
        s_r_feats: list,
        a_cfg_scale: float,
        r_cfg_scale: float,
        e_cfg_scale: float,
        emo: str,
        nfe: int,
        seed: int
    ) :
        """
        Generator that yields decoded frames one by one.
        Each yielded frame is a GPU tensor ready for conversion.
        """
        # Sample latents frame-by-frame
        for frame_latent in self.sample(
            data,
            a_cfg_scale=a_cfg_scale,
            r_cfg_scale=r_cfg_scale,
            e_cfg_scale=e_cfg_scale,
            emo=emo,
            nfe=nfe,
            seed=seed,
            s_r=s_r
        ):
            # Decode on separate stream
            with torch.cuda.stream(self.stream_decode):
                decoded_frame = self.decode_latent_into_image(
                    s_r=s_r,
                    s_r_feats=s_r_feats,
                    r_d=frame_latent
                )
                
                # Handle tuple returns
                if isinstance(decoded_frame, (tuple, list)):
                    decoded_frame = decoded_frame[0]
                
                # Ensure on GPU
                if not decoded_frame.is_cuda:
                    decoded_frame = decoded_frame.to(device=self.opt.rank, non_blocking=True)
            
            # Sync decode stream before yielding
            torch.cuda.current_stream(device=self.opt.rank).wait_stream(self.stream_decode)
            
            # Clone to default stream
            frame_on_default = decoded_frame.clone()
            
            yield frame_on_default
    
    def _tensor_to_numpy(self, frame_tensor: torch.Tensor) -> np.ndarray:
        """
        Convert a single frame tensor to numpy uint8 array.
        Handles GPU->CPU transfer efficiently.
        """
        # Ensure batch dimension [B=1, C, H, W]
        if frame_tensor.ndim == 3:  # [C, H, W]
            frame_tensor = frame_tensor.unsqueeze(0)
        
        # Permute to [B, H, W, C] on GPU
        frame_tensor = frame_tensor.permute(0, 2, 3, 1).contiguous()
        
        # Convert to uint8 on GPU
        if torch.is_floating_point(frame_tensor):
            frame_uint8_gpu = (frame_tensor.clamp(-1.0, 1.0) * 127.5 + 127.5).to(torch.uint8)
        else:
            frame_uint8_gpu = frame_tensor.to(torch.uint8)
        
        # Sync before CPU transfer
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Transfer to CPU and convert to numpy
        frame_np = frame_uint8_gpu.cpu().numpy()
        
        # Ensure contiguous
        if not frame_np.flags['C_CONTIGUOUS']:
            frame_np = np.ascontiguousarray(frame_np)
        
        return frame_np
    
    def _start_ffmpeg_process(
        self, 
        video_path: str, 
        height: int, 
        width: int, 
        channels: int, 
        fps: int
    ) -> subprocess.Popen:
        """
        Start FFmpeg process for streaming frame input.
        """
        pix_fmt_in = "rgba" if channels == 4 else "rgb24"
        size_str = f"{width}x{height}"
        
        base_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", pix_fmt_in,
            "-s", size_str,
            "-r", str(fps),
            "-i", "-",  # stdin
        ]
        
        encode_opts = [
            "-c:v", "h264_nvenc",
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
        
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=655360  # 64KB buffer
        )
        
        return proc

    
class FLOAT(BaseModel,StreamingVideoSaver):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.stream_decode = torch.cuda.Stream(device=self.opt.rank)  # device optional
        self.stream_encode= torch.cuda.Stream(device=self.opt.rank)  # device optional

        self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(self.opt.num_prev_frames)

        # motion latent auto-encoder
        self.motion_autoencoder = Generator(size = opt.input_size, style_dim = opt.dim_w, motion_dim = opt.dim_m)
        self.motion_autoencoder.requires_grad_(False)

        # condition encoders
        self.audio_encoder 		= AudioEncoder(opt)
        self.emotion_encoder	= Audio2Emotion(opt)
        self.decode_semaphore=asyncio.Semaphore(10)
        # FMT; Flow Matchingssss Transformer
        self.fmt = FlowMatchingTransformer(opt)
        
        # ODE options
        self.odeint_kwargs = {
            'atol': self.opt.ode_atol,
            'rtol': self.opt.ode_rtol,
            'method': self.opt.torchdiffeq_ode_method
        }
    stream = torch.cuda.Stream()
    ######## Motion Encoder - Decoder ########
    @torch.no_grad()
    def encode_image_into_latent(self, x: torch.Tensor) -> list:
        x_r, _, x_r_feats = self.motion_autoencoder.enc(x, input_target=None)
        x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
        return x_r, x_r_lambda, x_r_feats

    @torch.no_grad()
    def encode_identity_into_motion(self, x_r: torch.Tensor) -> torch.Tensor:
        x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
        r_x = self.motion_autoencoder.dec.direction(x_r_lambda)
        return r_x

    @torch.no_grad()
    def decode_latent_into_image(self, s_r: torch.Tensor, s_r_feats: list, r_d: torch.Tensor) -> torch.Tensor:
            """Decode a single frame (not a chunk with time dimension)"""
            # r_d is now [B, dim] - a single frame
            img, _ = self.motion_autoencoder.dec(
                r_d, 
                alpha=None, 
                feats=s_r_feats
            )
            return img  # Return single decoded frame


    @torch.no_grad()
    def sample(
        self,
        data: dict,
        a_cfg_scale: float = 1.0,
        r_cfg_scale: float = 1.0,
        e_cfg_scale: float = 1.0,
        emo: str = None,
        nfe: int = 10,
        seed: int = None,
        s_r=None
    ) -> torch.Tensor:
        with torch.cuda.stream(self.stream_encode):
            r_s, a = data['r_s'], data['a']
            B = a.shape[0]
            # make time 
            time = torch.linspace(0, 1, self.opt.nfe, device=self.opt.rank)
            
            # encoding audio first with whole audio
            a = a.to(self.opt.rank)
            T = math.ceil(a.shape[-1] * self.opt.fps / self.opt.sampling_rate)
            wa = self.audio_encoder.inference(a, seq_len=T)
            
            # encoding emotion first
            emo_idx = self.emotion_encoder.label2id.get(str(emo).lower(), None)
            if emo_idx is None:
                we = self.emotion_encoder.predict_emotion(a).unsqueeze(1)
            else:
                we = F.one_hot(torch.tensor(emo_idx, device=a.device), num_classes=self.opt.dim_e).unsqueeze(0).unsqueeze(0)
            
            decoded_length = 0
            
            # sampling chunk by chunk
            for t in range(0, int(math.ceil(T / self.num_frames_for_clip))):
                if self.opt.fix_noise_seed:
                    seed = self.opt.seed if seed is None else seed  
                    g = torch.Generator(self.opt.rank)
                    g.manual_seed(seed)
                    x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device=self.opt.rank, generator=g)
                else:
                    x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device=self.opt.rank)
                
                if t == 0:  # should define the previous
                    prev_x_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
                    prev_wa_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
                else:
                    prev_x_t = sample_t[:, -self.num_prev_frames:]
                    prev_wa_t = wa_t[:, -self.num_prev_frames:]
                
                wa_t = wa[:, t * self.num_frames_for_clip: (t+1) * self.num_frames_for_clip]
                if wa_t.shape[1] < self.num_frames_for_clip:  # padding by replicate
                    wa_t = F.pad(wa_t, (0, 0, 0, self.num_frames_for_clip - wa_t.shape[1]), mode='replicate')
                
                def sample_chunk(tt, zt):
                    out = self.fmt.forward_with_cfv(
                        t=tt.unsqueeze(0),
                        x=zt,
                        wa=wa_t,
                        wr=r_s,
                        we=we,
                        prev_x=prev_x_t,
                        prev_wa=prev_wa_t,
                        a_cfg_scale=a_cfg_scale,
                        r_cfg_scale=r_cfg_scale,
                        e_cfg_scale=e_cfg_scale
                    )
                    out_current = out[:, self.num_prev_frames:]
                    return out_current
                
                # solve ODE 
                trajectory_t = odeint(sample_chunk, x0, time, **self.odeint_kwargs)
                sample_t = trajectory_t[-1]
                
                # Trim to exact length T on the last chunk
                frames_remaining = T - decoded_length
                if frames_remaining < sample_t.shape[1]:
                    sample_t_trim = sample_t[:, :frames_remaining]
                else:
                    sample_t_trim = sample_t
                #sample_t_trim = sample_t
                samples_len = sample_t_trim.shape[1]
                decoded_length+=samples_len
                # Add reference and yield frame by frame
                s_r_d_t = s_r + sample_t_trim
                
                # Yield frame by frame for streaming
                #print("shape of sample_t before squeeze",s_r_d_t.shape)
                
                s_r_d_t_squeezed = s_r_d_t.squeeze(0)  # [T, C, H, W]
                T = s_r_d_t_squeezed.shape[0]

                # Divide into 3 parts
                part_size = T // 3
                remainder = T % 3

                # Split points
                split1 = part_size + (1 if remainder > 0 else 0)
                split2 = split1 + part_size + (1 if remainder > 1 else 0)

                # Yield each part
                yield s_r_d_t_squeezed[:split1]      # Part 1
                yield s_r_d_t_squeezed[split1:split2]  # Part 2
                yield s_r_d_t_squeezed[split2:]      # Part 3    
                #print("shape of sample_t after squeeze",s_r_d_t.squeeze(0).shape)

                # Stop if we've reached the target length
                if decoded_length >= T:
                    break

    @torch.no_grad()
    def inference(self, data: dict, a_cfg_scale=None, r_cfg_scale=None, 
                e_cfg_scale=None, emo=None, nfe=10, seed=None) -> dict:
        s, a = data['s'], data['a']
        s_r, r_s_lambda, s_r_feats = self.encode_image_into_latent(s.to(self.opt.rank))

        if 's_r' in data:
            r_s = self.encode_identity_into_motion(s_r)
        else:
            r_s = self.motion_autoencoder.dec.direction(r_s_lambda)
        data['r_s'] = r_s

        if a_cfg_scale is None: a_cfg_scale = self.opt.a_cfg_scale
        if r_cfg_scale is None: r_cfg_scale = self.opt.r_cfg_scale
        if e_cfg_scale is None: e_cfg_scale = self.opt.e_cfg_scale

        try:
            # list of (tensor_on_gpu, stream)
            decode_results = []

            #s_r_feats = [f.expand(50, -1, -1, -1) for f in s_r_feats]

            # If sample yields lazily, we will create per-frame streams and decode on them.
            for frame_latent in self.sample(data,
                                            a_cfg_scale=a_cfg_scale,
                                            r_cfg_scale=r_cfg_scale,
                                            e_cfg_scale=e_cfg_scale,
                                            emo=emo, nfe=nfe, seed=seed, s_r=s_r):
                # create and keep a reference to the stream
                # launch decode on that stream
                 with torch.cuda.stream(self.stream_decode):

                    # IMPORTANT: decode_latent_into_image must perform CUDA ops only,
                    # and must NOT move tensors to CPU / PIL or call .item() etc.
                    try:
                        out = self.decode_latent_into_image(s_r=s_r, s_r_feats=s_r_feats, r_d=frame_latent)
                        # out should be a torch.Tensor located on GPU (e.g. [B, C, H, W])
                        # If decode returns (tensor, other) adapt accordingly.
                        if isinstance(out, tuple) or isinstance(out, list):
                            # assume primary tensor is first element
                            out_tensor = out[0]
                        else:
                            out_tensor = out

                        # ensure it's a CUDA tensor
                        if not out_tensor.is_cuda:
                            # If your decode runs on CPU, you lose streaming benefits. Move to GPU (non-blocking)
                            out_tensor = out_tensor.to(device=self.opt.rank, non_blocking=True)

                        # keep pair for later synchronization/stack
                        decode_results.append((out_tensor, self.stream_decode))
                    except Exception as e:
                        print(f"Error: {e}\n{traceback.format_exc()}")
                        raise 

            # now wait for each stream to finish and move outputs to the current/default stream
            stacked_list = []
            cur = torch.cuda.current_stream(device=self.opt.rank)
            for out_tensor, stream in decode_results:
                # make sure default stream waits for worker stream before we touch `out_tensor`
                cur.wait_stream(stream)
                # Clone to default stream
                out_on_default = out_tensor.clone()  # shape: [T, C, H, W]
                print("shape of out_on_default", out_on_default.shape)
                stacked_list.append(out_on_default)

            # Concatenate along temporal dimension (dim=0): [T_total, C, H, W]
            d_hat = torch.cat(stacked_list, dim=0)  # [156, 3, 512, 512] in your example


            # If you really need [B, T, H, W, C], permute:
            print("d_hat",d_hat.shape)
            # if B==1 and you want remove batch dim:
            if d_hat.shape[1] == 1:
                d_hat = d_hat.squeeze(1)
        
            return {'d_hat': d_hat}

        except Exception as e:
            print(f"Error: {e}\n{traceback.format_exc()}")
            raise

@dataclass
class FrameData:
    """Container for frame data passed between producer and consumer"""
    frame_np: np.ndarray
    frame_idx: int
    is_last: bool = False


# Usage example:
"""
# The run_inference method now uses producer-consumer pattern internally:
result_path = model.run_inference(
    res_video_path='output.mp4',
    ref_path='reference.jpg',
    audio_path='audio.wav',
    a_cfg_scale=2.0,
    r_cfg_scale=1.0,
    e_cfg_scale=1.0,
    emo='happy',
    nfe=10,
    verbose=True
)
"""

################ Condition Encoders ################
class AudioEncoder(BaseModel):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.only_last_features = opt.only_last_features
        
        self.num_frames_for_clip = int(opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(opt.num_prev_frames)
        if os.path.exists(opt.wav2vec_model_path):

                self.wav2vec2 = Wav2VecModel.from_pretrained(opt.wav2vec_model_path, local_files_only = True)
        else:
                self.wav2vec2 = Wav2VecModel.from_pretrained(opt.wav2vec_model_path)

        self.wav2vec2.feature_extractor._freeze_parameters()

        for name, param in self.wav2vec2.named_parameters():
            param.requires_grad = False

        audio_input_dim = 768 if opt.only_last_features else 12 * 768

        self.audio_projection = nn.Sequential(
            nn.Linear(audio_input_dim, opt.dim_w),
            nn.LayerNorm(opt.dim_w),
            nn.SiLU()
            )

    def get_wav2vec2_feature(self, a: torch.Tensor, seq_len:int) -> torch.Tensor:
        a = self.wav2vec2(a, seq_len=seq_len, output_hidden_states = not self.only_last_features)
        if self.only_last_features:
            a = a.last_hidden_state
        else:
            a = torch.stack(a.hidden_states[1:], dim=1).permute(0, 2, 1, 3)
            a = a.reshape(a.shape[0], a.shape[1], -1)
        return a

    def forward(self, a:torch.Tensor, prev_a:torch.Tensor = None) -> torch.Tensor:
        if prev_a is not None:
            a = torch.cat([prev_a, a], dim = 1)
            if a.shape[1] % int( (self.num_frames_for_clip + self.num_prev_frames) * self.opt.sampling_rate / self.opt.fps) != 0:
                a = F.pad(a, (0, int((self.num_frames_for_clip + self.num_prev_frames) * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode='replicate')
            a = self.get_wav2vec2_feature(a, seq_len = self.num_frames_for_clip + self.num_prev_frames)
        else:
            if a.shape[1] % int( self.num_frames_for_clip * self.opt.sampling_rate / self.opt.fps) != 0:
                a = F.pad(a, (0, int(self.num_frames_for_clip * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode = 'replicate')
            a = self.get_wav2vec2_feature(a, seq_len = self.num_frames_for_clip)
    
        return self.audio_projection(a) # frame by frame

    @torch.no_grad()
    def inference(self, a: torch.Tensor, seq_len:int) -> torch.Tensor:
        if a.shape[1] % int(seq_len * self.opt.sampling_rate / self.opt.fps) != 0:
            a = F.pad(a, (0, int(seq_len * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode = 'replicate')
        a = self.get_wav2vec2_feature(a, seq_len=seq_len)
        return self.audio_projection(a)



class Audio2Emotion(nn.Module):
    def __init__(self, opt):
        super().__init__()

        # Load pretrained model (local preferred)
        if os.path.exists(opt.audio2emotion_path):
            self.wav2vec2_for_emotion = Wav2Vec2ForSpeechClassification.from_pretrained(
                opt.audio2emotion_path,
                local_files_only=True
            )
        else:
            self.wav2vec2_for_emotion = Wav2Vec2ForSpeechClassification.from_pretrained(
                opt.audio2emotion_path
            )

        self.wav2vec2_for_emotion.eval()

        # seven labels
        self.id2label = {
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "neutral",
            5: "sad",
            6: "surprise"
        }

        self.label2id = {v: k for k, v in self.id2label.items()}

    @torch.no_grad()
    def predict_emotion(self, a: torch.Tensor, prev_a: torch.Tensor = None) -> torch.Tensor:
        # Concatenate previous audio if provided
        if prev_a is not None:
            a = torch.cat([prev_a, a], dim=1)

        logits = self.wav2vec2_for_emotion(a).logits
        return F.softmax(logits, dim=1)