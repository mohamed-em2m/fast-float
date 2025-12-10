import os
import torch, math
import torch.nn as nn
import torch.nn.functional as F
import asyncio
from torchdiffeq import odeint
from transformers import Wav2Vec2Config
from transformers.modeling_outputs import BaseModelOutput

from models.wav2vec2 import Wav2VecModel
from models.wav2vec2_ser import Wav2Vec2ForSpeechClassification

from models import BaseModel
from models.float.generator import Generator
from models.float.FMT import FlowMatchingTransformer
import traceback
torch.backends.cudnn.benchmark = True
######## Main Phase 2 model ########		
class FLOAT(BaseModel):
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
								sample_t_trim=sample_t[:,:12]
								s_r_d_t = s_r + sample_t_trim
								samples_len = s_r_d_t.shape[1]
  
								# Yield frame by frame for streaming
								for sample_idx in range(samples_len):
									yield s_r_d_t[:,sample_idx]


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
					decode_results.append((out_tensor, self.stream))

			# now wait for each stream to finish and move outputs to the current/default stream
			stacked_list = []
			cur = torch.cuda.current_stream(device=self.opt.rank)
			for out_tensor, stream in decode_results:
				# make sure default stream waits for worker stream before we touch `out_tensor`
				# either synchronize the worker stream, or use wait_stream
				# Option A: wait on current stream (non-blocking) then clone to ensure safety:
				cur.wait_stream(stream)
				# Optionally clone to default stream so subsequent ops are on default stream:
				out_on_default = out_tensor.clone()  # clone schedules kernel on current/default stream
				stacked_list.append(out_on_default)

			# By now all clones are scheduled on the current stream; synchronize if you need CPU result
			# torch.cuda.synchronize(device=self.opt.rank)  # optional full sync

			# expected shape: [B, C, H, W] for each frame -> stack into [B, T, C, H, W]
			d_hat = torch.stack(stacked_list, dim=1)  # [B, T, C, H, W]

			# If you really need [B, T, H, W, C], permute:
			# d_hat = d_hat.permute(0, 1, 3, 4, 2)  # to [B, T, H, W, C]

			# if B==1 and you want remove batch dim:
			if d_hat.shape[0] == 1:
				d_hat = d_hat.squeeze(0)

			return {'d_hat': d_hat}

		except Exception as e:
			print(f"Error: {e}\n{traceback.format_exc()}")
			raise



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