"""Data preprocessing module for face landmark alignment, image transforms, and wav2vec audio feature extraction."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import albumentations as A
import albumentations.pytorch.transforms as A_pytorch
import cv2
import face_alignment
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor


class DataProcessor:
    """Preprocesses input reference face images and audio files for FLOAT inference."""

    def __init__(self, opt: Any) -> None:
        self.opt = opt
        self.fps = opt.fps
        self.sampling_rate = opt.sampling_rate
        self.input_size = opt.input_size

        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

        wav2vec_path = getattr(opt, "wav2vec_model_path", "")
        if wav2vec_path and os.path.exists(wav2vec_path):
            self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_path, local_files_only=True)
        elif wav2vec_path:
            self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_path)
        else:
            self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

        self.transform = A.Compose(
            [
                A.Resize(
                    height=opt.input_size,
                    width=opt.input_size,
                    interpolation=cv2.INTER_AREA,
                ),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                A_pytorch.ToTensorV2(),
            ]
        )
        self._ref_image_cache: dict[tuple[str, float, bool], torch.Tensor] = {}

    @torch.no_grad()
    def process_img(self, img: np.ndarray) -> np.ndarray:
        mult = 360.0 / img.shape[0]
        resized_img = cv2.resize(
            img,
            dsize=(0, 0),
            fx=mult,
            fy=mult,
            interpolation=cv2.INTER_AREA if mult < 1.0 else cv2.INTER_CUBIC,
        )
        bboxes = self.fa.face_detector.detect_from_image(resized_img)
        bboxes = [
            (int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), score)
            for (x1, y1, x2, y2, score) in bboxes
            if score > 0.95
        ]
        if not bboxes:
            raise ValueError("No face detected with confidence > 0.95. Use --no_crop.")
        bbox = bboxes[0]

        bsy = int((bbox[3] - bbox[1]) / 2)
        bsx = int((bbox[2] - bbox[0]) / 2)
        my = int((bbox[1] + bbox[3]) / 2)
        mx = int((bbox[0] + bbox[2]) / 2)
        bs = int(max(bsy, bsx) * 1.6)

        img = cv2.copyMakeBorder(img, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=0)
        my, mx = my + bs, mx + bs
        crop_img = img[my - bs : my + bs, mx - bs : mx + bs]
        crop_img = cv2.resize(
            crop_img,
            dsize=(self.input_size, self.input_size),
            interpolation=cv2.INTER_AREA if mult < 1.0 else cv2.INTER_CUBIC,
        )
        return crop_img

    def default_img_loader(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def default_aud_loader(self, path: str) -> torch.Tensor:
        speech_array, sampling_rate = librosa.load(path, sr=self.sampling_rate)
        return self.wav2vec_preprocessor(speech_array, sampling_rate=sampling_rate, return_tensors="pt").input_values[0]

    def preprocess(
        self,
        ref_path: str | Sequence[str],
        audio_path: str | Sequence[str],
        no_crop: bool = False,
    ) -> dict[str, Any]:
        if isinstance(ref_path, (list, tuple)):
            s_list = []
            for rp in ref_path:
                mtime = os.path.getmtime(rp) if isinstance(rp, str) and os.path.exists(rp) else 0.0
                cache_key = (rp, mtime, no_crop)
                if cache_key in self._ref_image_cache:
                    s_tensor = self._ref_image_cache[cache_key]
                else:
                    img = self.default_img_loader(rp)
                    if not no_crop:
                        img = self.process_img(img)
                    s_tensor = self.transform(image=img)["image"]
                    self._ref_image_cache[cache_key] = s_tensor
                s_list.append(s_tensor)
            s = torch.stack(s_list, dim=0)
        else:
            mtime = os.path.getmtime(ref_path) if isinstance(ref_path, str) and os.path.exists(ref_path) else 0.0
            cache_key = (ref_path, mtime, no_crop)
            if cache_key in self._ref_image_cache:
                s = self._ref_image_cache[cache_key].unsqueeze(0)
            else:
                s_img = self.default_img_loader(ref_path)
                if not no_crop:
                    s_img = self.process_img(s_img)
                s_t = self.transform(image=s_img)["image"]
                self._ref_image_cache[cache_key] = s_t
                s = s_t.unsqueeze(0)

        if isinstance(audio_path, (list, tuple)):
            a_list = [self.default_aud_loader(ap) for ap in audio_path]
            max_len = max(aud.shape[-1] for aud in a_list)
            a_padded = [
                (F.pad(aud, (0, max_len - aud.shape[-1]), mode="constant", value=0) if aud.shape[-1] < max_len else aud)
                for aud in a_list
            ]
            a = torch.stack(a_padded, dim=0)
        else:
            a = self.default_aud_loader(audio_path).unsqueeze(0)

        return {"s": s, "a": a, "p": None, "e": None}
