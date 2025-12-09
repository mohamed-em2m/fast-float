import os
import torch, math
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint
from transformers import Wav2Vec2Config
from transformers.modeling_outputs import BaseModelOutput
import os
import math
import torch
import torch.nn.functional as F
import concurrent.futures
from models.wav2vec2 import Wav2VecModel
from models.wav2vec2_ser import Wav2Vec2ForSpeechClassification

from models import BaseModel
from models.float.generator import Generator
from models.float.FMT import FlowMatchingTransformer

######## Main Phase 2 model ########		
import os
import math
import torch
import torch.nn.functional as F
import concurrent.futures

class FLOAT(BaseModel):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(self.opt.num_prev_frames)

        # motion latent auto-encoder
        self.motion_autoencoder = Generator(size = opt.input_size, style_dim = opt.dim_w, motion_dim = opt.dim_m)
        self.motion_autoencoder.requires_grad_(False)

        # condition encoders
        self.audio_encoder      = AudioEncoder(opt)
        self.emotion_encoder    = Audio2Emotion(opt)

        # FMT; Flow Matching Transformer
        self.fmt = FlowMatchingTransformer(opt)
        
        # ODE options
        self.odeint_kwargs = {
            'atol': self.opt.ode_atol,
            'rtol': self.opt.ode_rtol,
            'method': self.opt.torchdiffeq_ode_method
        }

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
    def decode_latent_into_image(self, s_r: torch.Tensor , s_r_feats: list, r_d: torch.Tensor) -> dict:
        T = r_d.shape[1]
        d_hat = []
        for t in range(T):
            s_r_d_t = s_r + r_d[:, t]
            img_t, _ = self.motion_autoencoder.dec(s_r_d_t, alpha = None, feats = s_r_feats)
            d_hat.append(img_t)
        d_hat = torch.stack(d_hat, dim=1).squeeze()
        return {'d_hat': d_hat}


    ######## Motion Sampling and Inference ########
    @torch.no_grad()
    def sample(
        self,
        data: dict,
        a_cfg_scale: float = 1.0,
        r_cfg_scale: float = 1.0,
        e_cfg_scale: float = 1.0,
        emo: str = None,
        nfe: int = 10,
        seed: int = None
    ) -> torch.Tensor:

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
            we = F.one_hot(torch.tensor(emo_idx, device = a.device), num_classes = self.opt.dim_e).unsqueeze(0).unsqueeze(0)

        samples_per_chunk = []
        num_chunks = int(math.ceil(T / self.num_frames_for_clip))

        # determine threadpool size (allow override via opt; otherwise reasonable default)
        default_workers = min(4, (os.cpu_count() or 1))
        max_workers = getattr(self.opt, 'num_thread_workers', default_workers)

        # Helper per-chunk processing function
        def _process_chunk(t_idx, seed_for_chunk):
            # Prepare z0 (x0) per-chunk (with deterministic seeding if requested)
            if self.opt.fix_noise_seed:
                g = torch.Generator(self.opt.rank)
                g.manual_seed(seed_for_chunk)
                x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device = self.opt.rank, generator = g)
            else:
                x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device = self.opt.rank)

            # previous frames
            if t_idx == 0:
                prev_x_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w, device=self.opt.rank)
                prev_wa_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w, device=self.opt.rank)
            else:
                # These variables must be taken from samples_per_chunk computed earlier.
                # To keep independent workers (and avoid race with incomplete previous chunks),
                # we gather prev frames from concatenated results of prior chunks in main thread,
                # so here we will return x0->trajectory and the main thread will manage prev_x_t for later chunks.
                # For correctness we keep this function self-contained by re-computing prev_x_t and prev_wa_t
                # from the already concatenated 'prefix' tensors passed below.
                # NOTE: this function will be called with the appropriate prefix state via closure in the main thread.
                pass

            # wa slice for this chunk
            wa_t_local = wa[:, t_idx * self.num_frames_for_clip: (t_idx + 1) * self.num_frames_for_clip]
            if wa_t_local.shape[1] < self.num_frames_for_clip: # padding by replicate
                wa_t_local = F.pad(wa_t_local, (0, 0, 0, self.num_frames_for_clip - wa_t_local.shape[1]), mode='replicate')

            # We'll close over prev_x_t and prev_wa_t supplied by the caller (see below)
            # Define sample_chunk for odeint (it accepts tt, zt)
            def sample_chunk(tt, zt):
                out = self.fmt.forward_with_cfv(
                    t           = tt.unsqueeze(0),
                    x           = zt,
                    wa          = wa_t_local,              
                    wr          = r_s,
                    we          = we, 
                    prev_x      = _process_chunk.prev_x_t,
                    prev_wa     = _process_chunk.prev_wa_t,
                    a_cfg_scale = a_cfg_scale,
                    r_cfg_scale = r_cfg_scale,
                    e_cfg_scale = e_cfg_scale
                )
                out_current = out[:, self.num_prev_frames:]
                return out_current

            # call odeint (heavy op)
            trajectory_t = odeint(sample_chunk, x0, time, **self.odeint_kwargs)
            sample_t = trajectory_t[-1]
            return sample_t

        # We'll keep a prefix concatenation of samples so future chunks can see previous frames
        concatenated_prefix = None
        wa_prefix = None

        # Use ThreadPoolExecutor to process chunks — but because later chunks depend on previous chunk outputs
        # (prev_x_t, prev_wa_t), we cannot blindly launch all chunks at once unless we capture the prefix state.
        # To balance concurrency and correctness, we submit chunks in small batches while ensuring each worker
        # has the prefix it needs. Here we do optimistic parallelism by executing up to `max_workers` chunks in parallel
        # where each chunk's prev state is taken from the currently available concatenated_prefix.
        chunk_idx = 0
        samples_chunks = [None] * num_chunks

        # If seeds should be deterministic, prepare them per chunk
        base_seed = self.opt.seed if getattr(self.opt, 'seed', None) is not None else 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # We'll iterate over chunks, submitting up to max_workers at a time.
            while chunk_idx < num_chunks:
                running = []
                # submit batch
                for j in range(max_workers):
                    if chunk_idx >= num_chunks:
                        break

                    # compute prev_x_t and prev_wa_t for this chunk from concatenated_prefix
                    if concatenated_prefix is None:
                        prev_x_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w, device=self.opt.rank)
                        prev_wa_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w, device=self.opt.rank)
                    else:
                        # extract last num_prev_frames from concatenated_prefix and wa_prefix
                        prev_x_t = concatenated_prefix[:, -self.num_prev_frames:].clone()
                        prev_wa_t = wa_prefix[:, -self.num_prev_frames:].clone()

                    # store these into the function object so worker can use them (thread-safe per submission)
                    _process_chunk.prev_x_t = prev_x_t
                    _process_chunk.prev_wa_t = prev_wa_t

                    seed_for_chunk = (base_seed + chunk_idx) if self.opt.fix_noise_seed else None

                    future = executor.submit(_process_chunk, chunk_idx, seed_for_chunk)
                    running.append((chunk_idx, future))
                    chunk_idx += 1

                # collect results for this batch in order of submission
                for idx, fut in running:
                    sample_t = fut.result()  # will block until that chunk is done
                    samples_chunks[idx] = sample_t

                    # update concatenated_prefix and wa_prefix for next chunks
                    if concatenated_prefix is None:
                        concatenated_prefix = sample_t
                    else:
                        concatenated_prefix = torch.cat([concatenated_prefix, sample_t], dim=1)

                    # update wa_prefix similarly
                    wa_t_for_idx = wa[:, idx * self.num_frames_for_clip: (idx + 1) * self.num_frames_for_clip]
                    if wa_t_for_idx.shape[1] < self.num_frames_for_clip:
                        wa_t_for_idx = F.pad(wa_t_for_idx, (0, 0, 0, self.num_frames_for_clip - wa_t_for_idx.shape[1]), mode='replicate')
                    if wa_prefix is None:
                        wa_prefix = wa_t_for_idx
                    else:
                        wa_prefix = torch.cat([wa_prefix, wa_t_for_idx], dim=1)

        # After all chunks processed, concatenate and slice to T frames
        sample = torch.cat(samples_chunks, dim=1)[:, :T]
        return sample

    @torch.no_grad()
    def inference(
        self,
        data: dict,
        a_cfg_scale = None,
        r_cfg_scale = None,
        e_cfg_scale = None,
        emo         = None,
        nfe         = 10,
        seed        = None,
    ) -> dict:

        s, a = data['s'], data['a']
        s_r, r_s_lambda, s_r_feats = self.encode_image_into_latent(s.to(self.opt.rank))
        if 's_r' in data:
            r_s = self.encode_identity_into_motion(s_r)
        else:
            r_s = self.motion_autoencoder.dec.direction(r_s_lambda)
        data['r_s'] = r_s

        # set conditions
        if a_cfg_scale is None: a_cfg_scale = self.opt.a_cfg_scale
        if r_cfg_scale is None: r_cfg_scale = self.opt.r_cfg_scale
        if e_cfg_scale is None: e_cfg_scale = self.opt.e_cfg_scale

        sample = self.sample(data, a_cfg_scale = a_cfg_scale, r_cfg_scale = r_cfg_scale, e_cfg_scale = e_cfg_scale, emo = emo, nfe = nfe, seed = seed)
        data_out = self.decode_latent_into_image(s_r = s_r, s_r_feats = s_r_feats, r_d = sample)
        return data_out





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