import logging
import os
import sys

import torch
import torch.nn.functional as F

try:
    pass
except Exception:
    sys.modules["torchvision"] = None
    sys.modules["torchvision.io"] = None

from transformers import Wav2Vec2Model
from transformers.modeling_outputs import BaseModelOutput

logger = logging.getLogger(__name__)

# transformers builds ``Wav2Vec2PositionalConvEmbedding.conv`` with
# ``nn.utils.parametrizations.weight_norm`` whenever torch exposes it
# (torch>=2.1) and with the legacy ``nn.utils.weight_norm`` otherwise. Mirror
# that exact check so the shim below agrees with whichever module got built.
PARAMETRIZED_WEIGHT_NORM = hasattr(getattr(torch.nn.utils, "parametrizations", None), "weight_norm")


def read_checkpoint_state_dict(model_dir) -> dict:
    """Best-effort read of a local checkpoint dir's torch weights.

    Returns ``{}`` when the directory has no torch weight file (e.g. a Hub id
    that ``from_pretrained`` resolves through the HF cache), so callers can use
    it unconditionally.
    """
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = os.path.join(str(model_dir), name)
        if not os.path.exists(path):
            continue
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            return load_file(path, device="cpu")
        return torch.load(path, map_location="cpu", weights_only=True)
    return {}


def restore_legacy_weight_norm(model, state_dict) -> int:
    """Re-attach legacy ``weight_g``/``weight_v`` tensors to parametrized weights.

    Checkpoints saved with the legacy ``nn.utils.weight_norm`` store the
    positional conv as ``weight_g`` (magnitude) / ``weight_v`` (direction).
    When transformers instead builds the module with the parametrization API
    (torch>=2.1) those two keys match nothing, so the loader reports them as
    unused and leaves ``parametrizations.weight.original0/1`` at their random
    init — the audio and emotion encoders then run on a random positional conv
    (no lip sync) while every other weight still loads fine.

    The magnitude is identified by element count (it only spans the output
    channels), independent of how torch ordered ``original0``/``original1``.
    No-op on torch<2.1 and when the checkpoint holds no such pair.
    """
    if not PARAMETRIZED_WEIGHT_NORM or not state_dict:
        return 0
    restored = 0
    for name, module in model.named_modules():
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is None or "weight" not in parametrizations:
            continue
        g_key = next((key for key in state_dict if key.endswith(f"{name}.weight_g")), None)
        if g_key is None:
            continue
        v_key = f"{g_key[: -len('weight_g')]}weight_v"
        if v_key not in state_dict:
            continue
        weight = parametrizations.weight
        first, second = weight.original0, weight.original1
        if first.numel() > second.numel():
            first, second = second, first
        magnitude, direction = state_dict[g_key], state_dict[v_key]
        if first.numel() != magnitude.numel() or second.numel() != direction.numel():
            logger.warning("weight_norm shim: element count mismatch at %s; left as-is", name)
            continue
        with torch.no_grad():
            first.copy_(magnitude.reshape(first.shape))
            second.copy_(direction.reshape(second.shape))
        restored += 1
    if restored:
        logger.info("weight_norm shim: restored %d legacy tensor(s)", restored)
    return restored


def load_pretrained_with_weight_norm(model_cls, model_dir, **kwargs):
    """``from_pretrained`` that also restores legacy ``weight_norm`` tensors."""
    model = model_cls.from_pretrained(model_dir, **kwargs)
    restore_legacy_weight_norm(model, read_checkpoint_state_dict(model_dir))
    return model


class Wav2VecModel(Wav2Vec2Model):
    """
    Wav2VecModel is a custom model class that extends the Wav2Vec2Model class from the transformers library.
    It inherits all the functionality of the Wav2Vec2Model and adds additional methods for feature extraction and encoding.
    ...

    Attributes:
        base_model (Wav2Vec2Model): The base Wav2Vec2Model object.

    Methods:
        forward(input_values, seq_len, attention_mask=None, mask_time_indices=None
        , output_attentions=None, output_hidden_states=None, return_dict=None):
            Forward pass of the Wav2VecModel.
            It takes input_values, seq_len, and other optional parameters as input and returns the output of the base model.

        feature_extract(input_values, seq_len):
            Extracts features from the input_values using the base model.

        encode(extract_features, attention_mask=None, mask_time_indices=None, output_attentions=None, output_hidden_states=None, return_dict=None):
            Encodes the extracted features using the base model and returns the encoded features.
    """

    def _encode_with_hidden_states(self, hidden_states):
        """Run the transformer encoder, collecting per-layer outputs.

        transformers>=5.x removed hidden-state collection from
        ``Wav2Vec2Encoder`` (it returns only ``last_hidden_state``), which
        this fork needs for the 12-layer audio feature stack. This loop
        reproduces the v4 encoder math (same submodules, same order;
        ``attention_mask`` is always None in this repo) with the v4
        hidden-states convention: entry 0 is the encoder input
        (post-projection), entries 1..12 are the layer outputs.
        """
        enc = self.encoder
        stable = type(enc).__name__ == "Wav2Vec2EncoderStableLayerNorm"
        position_embeddings = enc.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings.to(hidden_states.device)
        if not stable:
            hidden_states = enc.layer_norm(hidden_states)
        hidden_states = enc.dropout(hidden_states)
        all_hidden_states = [hidden_states]
        for layer in enc.layers:
            hidden_states = layer(hidden_states, attention_mask=None)
            if isinstance(hidden_states, (tuple, list)):
                hidden_states = hidden_states[0]
            all_hidden_states.append(hidden_states)
        if stable:
            hidden_states = enc.layer_norm(hidden_states)
        return hidden_states, all_hidden_states

    def forward(
        self,
        input_values,
        seq_len=None,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass of the Wav2Vec model.

        Args:
            self: The instance of the model.
            input_values: The input values (waveform) to the model.
            seq_len: The sequence length of the input values.
            attention_mask: Attention mask to be used for the model.
            mask_time_indices: Mask indices to be used for the model.
            output_attentions: If set to True, returns attentions.
            output_hidden_states: If set to True, returns hidden states.
            return_dict: If set to True, returns a BaseModelOutput instead of a tuple.

        Returns:
            The output of the Wav2Vec model.
        """
        # NOTE: the old `self.config.output_attentions = True` line is gone:
        # on transformers>=4.4x with sdpa kernels that assignment raises, and
        # nothing downstream consumed attentions anyway.
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        with torch.no_grad():
            extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        if seq_len is not None:
            extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states1 = self._mask_hidden_states(
            hidden_states,
            mask_time_indices=mask_time_indices,
            attention_mask=attention_mask,
        )

        hidden_states, all_hidden_states = self._encode_with_hidden_states(hidden_states1)

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, tuple(all_hidden_states))

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states),
            attentions=None,
        )

    def feature_extract(
        self,
        input_values,
        seq_len,
    ):
        """
        Extracts features from the input values and returns the extracted features.

        Parameters:
        input_values (torch.Tensor): The input values to be processed.
        seq_len (torch.Tensor): The sequence lengths of the input values.

        Returns:
        extracted_features (torch.Tensor): The extracted features from the input values.
        """
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        return extract_features

    def encode(
        self,
        extract_features,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Encodes the input features into the output space.

        Args:
            extract_features (torch.Tensor): The extracted features from the audio signal.
            attention_mask (torch.Tensor, optional): Attention mask to be used for padding.
            mask_time_indices (torch.Tensor, optional): Masked indices for the time dimension.
            output_attentions (bool, optional): If set to True, returns the attention weights.
            output_hidden_states (bool, optional): If set to True, returns all hidden states.
            return_dict (bool, optional): If set to True, returns a BaseModelOutput instead of the tuple.

        Returns:
            The encoded output features.
        """
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states,
            mask_time_indices=mask_time_indices,
            attention_mask=attention_mask,
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, *encoder_outputs[1:])
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def linear_interpolation(features, seq_len):
    """
    Transpose the features to interpolate linearly.

    Args:
        features (torch.Tensor): The extracted features to be interpolated.
        seq_len (torch.Tensor): The sequence lengths of the features.

    Returns:
        torch.Tensor: The interpolated features.
    """
    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=seq_len, align_corners=True, mode="linear")
    return output_features.transpose(1, 2)
