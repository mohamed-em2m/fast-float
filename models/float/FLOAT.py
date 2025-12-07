import os
import torch, math
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint
from transformers import Wav2Vec2Config
from transformers.modeling_outputs import BaseModelOutput

from models.wav2vec2 import Wav2VecModel
from models.wav2vec2_ser import Wav2Vec2ForSpeechClassification

from models import BaseModel
from models.float.generator import Generator
from models.float.FMT import FlowMatchingTransformer

######## Main Phase 2 model ########		
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
		self.audio_encoder 		= AudioEncoder(opt)
		self.emotion_encoder	= Audio2Emotion(opt)

		# FMT; Flow Matching Transformer
		self.fmt = FlowMatchingTransformer(opt)
		
		# ODE options
		self.odeint_kwargs = {
			'atol': self.opt.ode_atol,
			'rtol': self.opt.ode_rtol,
			'method': self.opt.torchdiffeq_ode_method
		}

		# Streaming state cache - dictionary to store multiple request states
		self.streaming_states = {}  # {request_id: state_dict}
	
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
		
		# Stack frames: keep batch dimension if B > 1, otherwise squeeze only time dimension
		d_hat = torch.stack(d_hat, dim=1)
		if d_hat.shape[0] == 1:  # Only squeeze batch if batch size is 1
			d_hat = d_hat.squeeze(0)
		
		return {'d_hat': d_hat}

	def reset_streaming_state(self, request_id: str = "temp"):
		"""Reset streaming state for a specific request ID"""
		if request_id in self.streaming_states:
			del self.streaming_states[request_id]

	def get_streaming_state(self, request_id: str = "temp") -> dict:
		"""Get current streaming state for a specific request ID"""
		return self.streaming_states.get(request_id, None)

	def set_streaming_state(self, state: dict, request_id: str = "temp"):
		"""Set streaming state for a specific request ID"""
		self.streaming_states[request_id] = state

	def clear_all_streaming_states(self):
		"""Clear all streaming states"""
		self.streaming_states.clear()

	######## Motion Sampling and Inference ########
	@torch.no_grad()
	def sample_streaming(
		self,
		data: dict,
		a_cfg_scale: float = 1.0,
		r_cfg_scale: float = 1.0,
		e_cfg_scale: float = 1.0,
		emo: str = None,
		nfe: int = 10,
		seed: int = None,
		chunk_idx: int = 0,
		is_first_chunk: bool = True,
		return_state: bool = True,
		request_id: str = "temp"
	) -> tuple:
		"""
		Streaming generation that supports continuation from previous chunks
		This processes THE ENTIRE audio chunk, which may contain multiple clips
		
		Args:
			data: dict with 'r_s' and 'a' (audio chunk - can be any length)
			chunk_idx: starting chunk index for this audio segment
			is_first_chunk: whether this is the very first chunk of the entire sequence
			return_state: whether to return state for next chunk
			request_id: unique identifier for this generation request (default: "temp")
		
		Returns:
			sample: generated motion for entire audio chunk
			state: state dict for next chunk (if return_state=True)
		"""

		r_s, a = data['r_s'], data['a']
		B = a.shape[0]

		# make time 
		time = torch.linspace(0, 1, nfe, device=self.opt.rank)
		
		# encoding audio for this chunk
		a = a.to(self.opt.rank)
		T = math.ceil(a.shape[-1] * self.opt.fps / self.opt.sampling_rate)
		
		# Encode entire audio chunk
		wa = self.audio_encoder.inference(a, seq_len=T)

		# Get cached state for this request
		cached_state = self.streaming_states.get(request_id, None)

		# encoding emotion (reuse if cached)
		if is_first_chunk or cached_state is None:
			emo_idx = self.emotion_encoder.label2id.get(str(emo).lower(), None)
			if emo_idx is None:
				we = self.emotion_encoder.predict_emotion(a).unsqueeze(1)
			else:
				we = F.one_hot(torch.tensor(emo_idx, device = a.device), num_classes = self.opt.dim_e).unsqueeze(0).unsqueeze(0)
		else:
			# Reuse emotion from cached state
			we = cached_state['we']

		# Load previous frame states or initialize
		if is_first_chunk or cached_state is None:
			prev_x_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
			prev_wa_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
			current_chunk_idx = 0
		else:
			# Load from saved state - take only the minimum needed frames
			saved_prev_x = cached_state['prev_x']
			saved_prev_wa = cached_state['prev_wa']
			
			# If saved state has more frames than needed, take only the last N
			if saved_prev_x.shape[1] > self.num_prev_frames:
				prev_x_t = saved_prev_x[:, -self.num_prev_frames:]
				prev_wa_t = saved_prev_wa[:, -self.num_prev_frames:]
			# If saved state has fewer frames, pad with zeros
			elif saved_prev_x.shape[1] < self.num_prev_frames:
				pad_size = self.num_prev_frames - saved_prev_x.shape[1]
				prev_x_t = F.pad(saved_prev_x, (0, 0, pad_size, 0), mode='constant', value=0)
				prev_wa_t = F.pad(saved_prev_wa, (0, 0, pad_size, 0), mode='constant', value=0)
			else:
				prev_x_t = saved_prev_x
				prev_wa_t = saved_prev_wa
			
			current_chunk_idx = cached_state['chunk_idx'] + 1

		# Process audio chunk by chunk (loop through clips)
		samples = []
		num_clips = int(math.ceil(T / self.num_frames_for_clip))
		
		for clip_idx in range(num_clips):
			# Generate noise for this clip
			if self.opt.fix_noise_seed:
				current_seed = self.opt.seed if seed is None else seed
				clip_seed = current_seed + current_chunk_idx + clip_idx
				g = torch.Generator(self.opt.rank)
				g.manual_seed(clip_seed)
				x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device=self.opt.rank, generator=g)
			else:
				x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device=self.opt.rank)

			# Get audio features for this clip
			wa_t = wa[:, clip_idx * self.num_frames_for_clip: (clip_idx + 1) * self.num_frames_for_clip]
			
			if wa_t.shape[1] < self.num_frames_for_clip:  # Pad last clip if needed
				wa_t = F.pad(wa_t, (0, 0, 0, self.num_frames_for_clip - wa_t.shape[1]), mode='replicate')

			def sample_clip(tt, zt):
				out = self.fmt.forward_with_cfv(
						t 			= tt.unsqueeze(0),
						x 			= zt,
						wa 			= wa_t, 			 
						wr 			= r_s,
						we 			= we, 
						prev_x 		= prev_x_t, 	
						prev_wa 	= prev_wa_t,
						a_cfg_scale = a_cfg_scale,
						r_cfg_scale = r_cfg_scale,
						e_cfg_scale = e_cfg_scale
						)

				out_current = out[:, self.num_prev_frames:]
				return out_current

			# Solve ODE for this clip
			trajectory_t = odeint(sample_clip, x0, time, **self.odeint_kwargs)
			sample_t = trajectory_t[-1]
			samples.append(sample_t)
			
			# Update previous states for next clip - take only last num_prev_frames
			if sample_t.shape[1] >= self.num_prev_frames:
				prev_x_t = sample_t[:, -self.num_prev_frames:]
			else:
				# If generated sample is shorter than num_prev_frames, keep what we have
				prev_x_t = sample_t
			
			if wa_t.shape[1] >= self.num_prev_frames:
				prev_wa_t = wa_t[:, -self.num_prev_frames:]
			else:
				prev_wa_t = wa_t

		# Concatenate all clips and trim to actual length
		sample = torch.cat(samples, dim=1)[:, :T]

		# Save state for next chunk - only save minimum needed frames
		if return_state:
			# Determine how many frames to save (minimum of actual frames or num_prev_frames)
			frames_to_save = min(sample.shape[1], self.num_prev_frames)
			wa_frames_to_save = min(wa.shape[1], self.num_prev_frames)
			
			self.streaming_states[request_id] = {
				'prev_x': sample[:, -frames_to_save:].clone(),
				'prev_wa': wa[:, -wa_frames_to_save:].clone(),
				'we': we,
				'chunk_idx': current_chunk_idx + num_clips - 1
			}
			return sample, self.streaming_states[request_id]
		
		return sample

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
		"""Original full-sequence generation (non-streaming)"""

		r_s, a = data['r_s'], data['a']
		B = a.shape[0]

		# make time 
		time = torch.linspace(0, 1, nfe, device=self.opt.rank)
		
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

		sample = []
		# sampleing chunk by chunk
		for t in range(0, int(math.ceil(T / self.num_frames_for_clip))):
			if self.opt.fix_noise_seed:
				seed = self.opt.seed if seed is None else seed	
				g = torch.Generator(self.opt.rank)
				g.manual_seed(seed)
				x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device = self.opt.rank, generator = g)
			else:
				x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device = self.opt.rank)

			if t == 0: # should define the previous
				prev_x_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
				prev_wa_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
			else:
				prev_x_t = sample_t[:, -self.num_prev_frames:]
				prev_wa_t = wa_t[:, -self.num_prev_frames:]
			
			wa_t = wa[:, t * self.num_frames_for_clip: (t+1)*self.num_frames_for_clip]

			if wa_t.shape[1] < self.num_frames_for_clip: # padding by replicate
				wa_t = F.pad(wa_t, (0, 0, 0, self.num_frames_for_clip - wa_t.shape[1]), mode='replicate')

			def sample_chunk(tt, zt):
				out = self.fmt.forward_with_cfv(
						t 			= tt.unsqueeze(0),
						x 			= zt,
						wa 			= wa_t, 			 
						wr 			= r_s,
						we 			= we, 
						prev_x 		= prev_x_t, 	
						prev_wa 	= prev_wa_t,
						a_cfg_scale = a_cfg_scale,
						r_cfg_scale = r_cfg_scale,
						e_cfg_scale = e_cfg_scale
						)

				out_current = out[:, self.num_prev_frames:]
				return out_current

			# solve ODE
			trajectory_t = odeint(sample_chunk, x0, time, **self.odeint_kwargs)
			sample_t = trajectory_t[-1]
			sample.append(sample_t)
		sample = torch.cat(sample, dim=1)[:, :T]
		return sample

	@torch.no_grad()
	def inference(
		self,
		data: dict,
		a_cfg_scale = None,
		r_cfg_scale = None,
		e_cfg_scale = None,
		emo			= None,
		nfe			= 10,
		seed		= None,
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

	@torch.no_grad()
	def inference_streaming(
		self,
		data: dict,
		a_cfg_scale = None,
		r_cfg_scale = None,
		e_cfg_scale = None,
		emo			= None,
		nfe			= 10,
		seed		= None,
		chunk_idx	= 0,
		is_first_chunk = True,
		request_id: str = "temp",
		debug: bool = False
	) -> dict:
		"""
		Streaming inference for one audio chunk
		
		Args:
			data: dict with 's' (source image, only needed for first chunk), 'a' (audio chunk)
			chunk_idx: current chunk index
			is_first_chunk: whether this is the first chunk
			request_id: unique identifier for this generation request (default: "temp")
			debug: print debug information
		
		Returns:
			dict with generated frames and streaming state
		"""

		# Create cache key for this request
		cache_key = f"_cached_{request_id}"

		# Encode source image only on first chunk
		if is_first_chunk:
			s = data['s']
			s_r, r_s_lambda, s_r_feats = self.encode_image_into_latent(s.to(self.opt.rank))
			if 's_r' in data:
				r_s = self.encode_identity_into_motion(s_r)
			else:
				r_s = self.motion_autoencoder.dec.direction(r_s_lambda)
			
			if debug:
				print(f"[DEBUG] First chunk - Encoded source image")
				print(f"  s_r shape: {s_r.shape}, range: [{s_r.min():.3f}, {s_r.max():.3f}]")
				print(f"  r_s shape: {r_s.shape}, range: [{r_s.min():.3f}, {r_s.max():.3f}]")
			
			# Cache these for subsequent chunks with request_id
			setattr(self, f"{cache_key}_s_r", s_r)
			setattr(self, f"{cache_key}_s_r_feats", s_r_feats)
			setattr(self, f"{cache_key}_r_s", r_s)
		else:
			# Reuse cached values for this request_id
			s_r = getattr(self, f"{cache_key}_s_r")
			s_r_feats = getattr(self, f"{cache_key}_s_r_feats")
			r_s = getattr(self, f"{cache_key}_r_s")
			
			if debug:
				print(f"[DEBUG] Continuing chunk {chunk_idx} - Using cached features")

		data['r_s'] = r_s

		# set conditions
		if a_cfg_scale is None: a_cfg_scale = self.opt.a_cfg_scale
		if r_cfg_scale is None: r_cfg_scale = self.opt.r_cfg_scale
		if e_cfg_scale is None: e_cfg_scale = self.opt.e_cfg_scale

		if debug:
			print(f"[DEBUG] CFG scales - audio: {a_cfg_scale}, reference: {r_cfg_scale}, emotion: {e_cfg_scale}")

		# Generate motion for this chunk
		sample, state = self.sample_streaming(
			data, 
			a_cfg_scale=a_cfg_scale, 
			r_cfg_scale=r_cfg_scale, 
			e_cfg_scale=e_cfg_scale, 
			emo=emo, 
			nfe=nfe, 
			seed=seed,
			chunk_idx=chunk_idx,
			is_first_chunk=is_first_chunk,
			return_state=True,
			request_id=request_id
		)

		if debug:
			print(f"[DEBUG] Generated sample shape: {sample.shape}, range: [{sample.min():.3f}, {sample.max():.3f}]")

		# Decode to images
		data_out = self.decode_latent_into_image(s_r=s_r, s_r_feats=s_r_feats, r_d=sample)
		
		if debug:
			d_hat = data_out['d_hat']
			print(f"[DEBUG] Decoded frames shape: {d_hat.shape}, range: [{d_hat.min():.3f}, {d_hat.max():.3f}]")
			print(f"[DEBUG] Frame dtype: {d_hat.dtype}")
		
		data_out['streaming_state'] = state
		data_out['request_id'] = request_id
		
		return data_out

	def clear_streaming_cache(self, request_id: str = "temp"):
		"""Clear cached source image features for a specific request"""
		cache_key = f"_cached_{request_id}"
		
		if hasattr(self, f"{cache_key}_s_r"):
			delattr(self, f"{cache_key}_s_r")
			delattr(self, f"{cache_key}_s_r_feats")
			delattr(self, f"{cache_key}_r_s")
		
		self.reset_streaming_state(request_id)

	def clear_all_caches(self):
		"""Clear all cached data and streaming states"""
		# Clear all cached attributes
		attrs_to_delete = [attr for attr in dir(self) if attr.startswith('_cached_')]
		for attr in attrs_to_delete:
			delattr(self, attr)
		
		# Clear all streaming states
		self.clear_all_streaming_states()

	@torch.no_grad()
	def test_inference_pipeline(self, data: dict, request_id: str = "test") -> dict:
		"""
		Test the full inference pipeline and return diagnostic info
		Useful for debugging black image issues
		"""
		print("="*50)
		print("Testing Inference Pipeline")
		print("="*50)
		
		# Test encoding
		s = data['s'].to(self.opt.rank)
		print(f"\n1. Input image shape: {s.shape}")
		print(f"   Input range: [{s.min():.3f}, {s.max():.3f}]")
		
		s_r, r_s_lambda, s_r_feats = self.encode_image_into_latent(s)
		print(f"\n2. Encoded latent (s_r) shape: {s_r.shape}")
		print(f"   s_r range: [{s_r.min():.3f}, {s_r.max():.3f}]")
		
		r_s = self.motion_autoencoder.dec.direction(r_s_lambda)
		print(f"\n3. Motion direction (r_s) shape: {r_s.shape}")
		print(f"   r_s range: [{r_s.min():.3f}, {r_s.max():.3f}]")
		
		# Test with zero motion (should reconstruct original)
		print(f"\n4. Testing reconstruction with zero motion...")
		zero_motion = torch.zeros(1, 1, self.opt.dim_w).to(self.opt.rank)
		recon = self.decode_latent_into_image(s_r, s_r_feats, zero_motion)
		recon_img = recon['d_hat']
		print(f"   Reconstructed shape: {recon_img.shape}")
		print(f"   Reconstructed range: [{recon_img.min():.3f}, {recon_img.max():.3f}]")
		
		# Test with small random motion
		print(f"\n5. Testing with small random motion...")
		small_motion = torch.randn(1, 5, self.opt.dim_w).to(self.opt.rank) * 0.1
		motion_result = self.decode_latent_into_image(s_r, s_r_feats, small_motion)
		motion_img = motion_result['d_hat']
		print(f"   Motion result shape: {motion_img.shape}")
		print(f"   Motion result range: [{motion_img.min():.3f}, {motion_img.max():.3f}]")
		
		# Test audio encoding
		if 'a' in data:
			a = data['a'].to(self.opt.rank)
			T = math.ceil(a.shape[-1] * self.opt.fps / self.opt.sampling_rate)
			wa = self.audio_encoder.inference(a, seq_len=min(T, 10))
			print(f"\n6. Audio encoding shape: {wa.shape}")
			print(f"   Audio features range: [{wa.min():.3f}, {wa.max():.3f}]")
		
		print("\n" + "="*50)
		print("Pipeline test complete!")
		print("="*50)
		
		return {
			'input': s,
			'encoded': s_r,
			'reconstructed': recon_img,
			'with_motion': motion_img
		}


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