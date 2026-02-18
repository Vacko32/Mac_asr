"""
This file contains encoder module implementation for Whisper Large V3.
This file with the gpu implmentation was generated with the help of Kimi2-5. 
Author: Martin Vaculik (xvaculm00@stud.fit.vutbr.cz)
"""

import torch
import torch.nn as nn
import torchaudio
import math

from transformers.models.whisper.modeling_whisper import WhisperModel
from transformers.models.whisper.feature_extraction_whisper import (
    WhisperFeatureExtractor,
)
import numpy as np

"""
This file contains encoder module implementation 
"""


class WhisperEncoderLarge(nn.Module):
    """Whisper Large V3 encoder for audio feature extraction with GPU-accelerated feature extraction."""

    MODEL_NAME = "openai/whisper-large-v3"

    # Whisper feature extraction parameters
    N_FFT = 400
    HOP_LENGTH = 160
    N_MELS = 128  
    SAMPLE_RATE = 16000
    MAX_SECONDS = 30
    MAX_SAMPLES = SAMPLE_RATE * MAX_SECONDS
    EXPECTED_FRAMES = 3000  

    def __init__(self, config=None, device=None, dtype=None):
        super().__init__()
        target_device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        target_dtype = dtype if dtype is not None else torch.bfloat16

        ### Load Whisper model
        whisper_model = WhisperModel.from_pretrained(self.MODEL_NAME)
        self.encoder = whisper_model.encoder
        self.encoder.to(device=target_device, dtype=target_dtype)
        self.target_device = target_device
        self.target_dtype = target_dtype
        self.emb_dim = self.encoder.config.hidden_size

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.MODEL_NAME
        )

        self._setup_mel_filters()

    def _setup_mel_filters(self):
        """Setup mel filterbank using Whisper's actual mel filters."""
        mel_filters_np = self.feature_extractor.mel_filters

        mel_filters = torch.from_numpy(mel_filters_np).to(
            device=self.target_device, dtype=torch.float32
        )

        self.register_buffer("mel_filters", mel_filters)
        self.register_buffer(
            "window", torch.hann_window(self.N_FFT, device=self.target_device)
        )

    def _compute_log_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute log-mel spectrogram on GPU using Whisper's exact method.

        Args:
            audio: Tensor of shape (batch_size, num_samples) on GPU

        Returns:
            Log-mel spectrogram of shape (batch_size, n_mels, time_frames)
        """
        batch_size = audio.shape[0]

        if audio.shape[-1] < self.MAX_SAMPLES:
            pad_length = self.MAX_SAMPLES - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, pad_length))
        else:
            audio = audio[:, : self.MAX_SAMPLES]

        stft = torch.stft(
            audio,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            win_length=self.N_FFT,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # power spectogram
        magnitudes = stft.abs() ** 2

        # mel filterbank
        mel_spec = torch.matmul(self.mel_filters.T, magnitudes)

        # slice to exactly EXPECTED_FRAMES
        if mel_spec.shape[-1] > self.EXPECTED_FRAMES:
            mel_spec = mel_spec[:, :, : self.EXPECTED_FRAMES]

        # Convert to log scale with Whispe norm
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        max_vals = log_spec.max(dim=-1, keepdim=True)[0]
        log_spec = torch.maximum(log_spec, max_vals - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def forward(self, input_features, attention_mask=None, wav_mask=None):
        """
        Forward pass with GPU-accelerated feature extraction.

        Args:
            input_features: Raw audio waveforms as torch.Tensor (batch_size, num_samples)
                           or pre-computed mel features (batch_size, n_mels, time)
            attention_mask: Not used, kept for API compatibility
            wav_mask: Optional mask for valid audio positions
        """
        if isinstance(input_features, torch.Tensor):
            # Check if input is raw audio (1D or 2D with varying lengths) or pre-computed features
            if (
                input_features.dim() == 2
                and input_features.shape[-1] <= self.MAX_SAMPLES
            ):
                # Raw audio waveforms - process on GPU
                # Ensure audio is on the correct device
                audio = input_features.to(self.target_device, dtype=torch.float32)

                # Compute mel-spectrogram on GPU
                mel_features = self._compute_log_mel_spectrogram(audio)

                # Convert to encoder dtype
                mel_features = mel_features.to(self.target_dtype)

                # Whisper expects (batch, n_mels, time) but returns in that format already
                # No need to transpose as Whisper encoder expects (batch, n_mels, seq_len)
            else:
                # Assume pre-computed mel features
                mel_features = input_features.to(
                    self.target_device, dtype=self.target_dtype
                )
        elif isinstance(input_features, np.ndarray):
            # Fallback for numpy arrays - convert to tensor
            audio = torch.from_numpy(input_features).to(
                self.target_device, dtype=torch.float32
            )
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            mel_features = self._compute_log_mel_spectrogram(audio)
            mel_features = mel_features.to(self.target_dtype)
        else:
            raise ValueError(f"Unsupported input type: {type(input_features)}")

        # Run encoder
        encoder_outputs = self.encoder(mel_features, attention_mask=None)
        hidden_states = encoder_outputs.last_hidden_state
        batch_size, seq_len, _ = hidden_states.shape

        # Whisper does not support attention mask, return only ones
        attention_mask = torch.ones(
            batch_size, seq_len, dtype=torch.bool, device=self.target_device
        )
        return hidden_states, attention_mask
