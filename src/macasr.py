import torch
import logging
import math
from typing import Optional, Tuple
from torch import Tensor
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from einops import rearrange, repeat
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma3ForCausalLM,
)
from encoder import WhisperEncoderLarge
from llm import LLMFactory
from connector import ConnectorFactory

import logging

logger = logging.getLogger(__name__)


class DownSamplerOptimized:
    def __init__(self, epsilon: int = 4, device=None, dtype=None):
        self.epsilon = epsilon
        self.pattern = "b (n k) d -> b n (k d)"

    @torch.compile(mode="max-autotune", fullgraph=True)
    def downsample(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        N = S // self.epsilon
        trim_len = N * self.epsilon
        x = x[:, :trim_len]
        mask = mask[:, :trim_len]

        new_shape = (B, N, self.epsilon * D)
        downsampled = x.reshape(new_shape)

        # Becuase whisper does not support attention mask.
        # We will return ones as attention mask
        downsampled_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)

        return downsampled, downsampled_mask


class MacAsr(torch.nn.Module):
    def __init__(
        self,
        epsilon=4,
        device=None,
        dtype=None,
        freeze_encoder=True,
        freeze_llm=True,
        use_lora=False,
        encoder_name="wavlm",
        model="qwen",
        connector_type="linear",
        num_of_beams=1,
    ):
        super().__init__()

        self.encoder = WhisperEncoderLarge(device=device, dtype=dtype)
        self.emb_dim = self.encoder.emb_dim
        self.DownSampler = DownSamplerOptimized(
            epsilon, device=device, dtype=torch.bfloat16
        )

        self.model = LLMFactory.create(model)
        self.model.set_num_of_beams(num_of_beams)

        self.connector = ConnectorFactory.create(
            connector_type=connector_type,
            input_dim=self.emb_dim * epsilon,
            output_dim=self.model.hidden_size,
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        if freeze_llm:
            for p in self.model.model.parameters():
                p.requires_grad = False

        self.model.to(device)
        self.connector.to(device)
        self.to(device)

    def forward(self, x: torch.Tensor, wav=None, wav_mask=None, text_mask=None):
        y, mask = self.encoder(wav)
        down_tuple = self.DownSampler.downsample(y, mask)
        y_upscaled = self.connector(down_tuple[0])
        return self.model.forward(x, speech_emb=y_upscaled)

    def generate(self, wav=None, wav_mask=None):
        y, mask = self.encoder(wav)
        down_tuple = self.DownSampler.downsample(y, mask)
        y_upscaled = self.connector(down_tuple[0])
        return self.model.generate(speech_emb=y_upscaled)
