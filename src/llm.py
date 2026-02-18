"""
This file contains LLM implementations.
Author: Martin Vaculik (xvaculm00@stud.fit.vutbr.cz)
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from abc import ABC, abstractmethod
from peft import LoraConfig, get_peft_model
from einops import repeat


class LoRAWrapper:
    """Generic LoRA wrapper for LLM models."""

    def __init__(
        self,
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    ):
        self.config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(model, self.config)

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()


class LLM(ABC, nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        self.model = model_id
        self.PROMPT_TEXT_1 = "Transcribe the following audio to text.\n\nAudio:"
        self.PROMPT_TEXT_2 = "\n\nTranscription:"
        self.last_token_id = None  # this token is the last token in the prompt2


    @abstractmethod
    def forward(self, x: torch.Tensor, speech_emb: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def generate(self, speech_emb: torch.Tensor) -> torch.Tensor:
        pass
    


### LLM Implementations


class Qwen(LLM):
    def __init__(self, model_id: str, use_lora: bool = False):
        super().__init__(model_id)
        model_id = "Qwen/Qwen3-1.7B"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)
        self.hidden_size = self.model.config.hidden_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")

        if use_lora:
            self.lora = LoRAWrapper(
                self.model, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            self.model = self.lora.model
            self.lora.print_trainable_parameters()

        self.emb_cast = self.model.get_input_embeddings()

        with torch.no_grad():
            self.prompt1 = self.emb_cast(
                self.tokenizer(self.PROMPT_TEXT_1, return_tensors="pt").input_ids.to(
                    self.device
                )
            )
            # concatenate prompt2 without last token because it is given during training
            self.prompt2 = self.emb_cast(
                self.tokenizer(self.PROMPT_TEXT_2, return_tensors="pt")
                .input_ids[:, :-1]
                .to(self.device)
            )

        self.pattern = "b s d -> b (s d)"
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token

        x = self.tokenizer(self.PROMPT_TEXT_2, return_tensors="pt").input_ids
        self.last_token_id = x[:, -1].item()  # set last token id for shift

    def forward(self, x: torch.Tensor, speech_emb: torch.Tensor) -> torch.Tensor:
        prompt1 = repeat(self.prompt1, "1 s d -> b s d", b=speech_emb.shape[0])
        prompt2 = repeat(self.prompt2, "1 s d -> b s d", b=speech_emb.shape[0])
        self.p_c = prompt1.shape[1] + speech_emb.shape[1] + prompt2.shape[1]

        z = torch.cat([prompt1, speech_emb, prompt2, self.emb_cast(x)], dim=1)

        return self.model(inputs_embeds=z)

    def generate(self, speech_emb: torch.Tensor, wav_mask=None) -> torch.Tensor:
        B = speech_emb.shape[0]
        prompt1 = repeat(self.prompt1, "1 s d -> b s d", b=B)
        prompt2 = repeat(self.prompt2, "1 s d -> b s d", b=B)

        # Add the last_token_id (colon) to match training input structure
        # Training sees: [prompt1][speech][prompt2_no_colon][colon][text]
        # Generation must start with: [prompt1][speech][prompt2_no_colon][colon]
        start_token_emb = self.emb_cast(
            torch.tensor([[self.last_token_id]], device=speech_emb.device)
        )
        start_token_emb = repeat(start_token_emb, "1 1 d -> b 1 d", b=B)
        z = torch.cat([prompt1, speech_emb, prompt2, start_token_emb], dim=1)

        self.p_c = z.shape[1]  # Update p_c to include start token

        attention_mask = torch.ones(B, z.shape[1], dtype=torch.long, device=z.device)

        generate_ids = self.model.generate(
            inputs_embeds=z,
            attention_mask=attention_mask,
            max_new_tokens=128,
            num_beams=self.num_of_beams,
            do_sample=False,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.eos_token_id,
        )
        dec = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return dec

    def set_num_of_beams(self, num_of_beams: int):
        self.num_of_beams = num_of_beams


class Gemma(LLM):
    def __init__(self, model_id: str = "google/gemma-3-1b-pt", use_lora: bool = False):
        super().__init__(model_id)
        raise NotImplementedError("Gemma is not supported")

    def set_num_of_beams(self, num_of_beams: int):
        raise NotImplementedError("Gemma is not supported")


class GenericLLM(LLM):
    def __init__(self, model_id: str, use_lora: bool = False):
        super().__init__(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
        self.hidden_size = self.model.config.hidden_size
        self.emb_cast = self.model.get_input_embeddings()

        with torch.no_grad():
            self.prompt1 = self.emb_cast(
                self.tokenizer(self.PROMPT_TEXT_1, return_tensors="pt").input_ids.to(
                    self.device
                )
            )
            self.prompt2 = self.emb_cast(
                self.tokenizer(self.PROMPT_TEXT_2, return_tensors="pt")
                .input_ids[:, :-1]
                .to(self.device)
            )

        if use_lora:
            self.lora = LoRAWrapper(
                self.model, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            self.model = self.lora.model
            self.lora.print_trainable_parameters()

        self.pattern = "b s d -> b (s d)"
        x = self.tokenizer(self.PROMPT_TEXT_2, return_tensors="pt").input_ids
        self.last_token_id = x[:, -1].item()  # set last token id for shift

        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token

        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.eos_token_id
        )
        if self.tokenizer.pad_token is None:
            raise ValueError("PAD token is None, pad cannot be eos")
        self.pad_token = self.tokenizer.pad_token
        self.tokenizer.pad_token = self.pad_token

    def forward(self, x: torch.Tensor, speech_emb: torch.Tensor) -> torch.Tensor:
        prompt1 = repeat(self.prompt1, "1 s d -> b s d", b=speech_emb.shape[0])
        prompt2 = repeat(self.prompt2, "1 s d -> b s d", b=speech_emb.shape[0])
        self.p_c = prompt1.shape[1] + speech_emb.shape[1] + prompt2.shape[1]

        z = torch.cat([prompt1, speech_emb, prompt2, self.emb_cast(x)], dim=1)

        return self.model(inputs_embeds=z)

    def generate(self, speech_emb: torch.Tensor, wav_mask=None) -> torch.Tensor:
        B = speech_emb.shape[0]
        prompt1 = repeat(self.prompt1, "1 s d -> b s d", b=B)
        prompt2 = repeat(self.prompt2, "1 s d -> b s d", b=B)

        # Add the last_token_id (colon) to match training input structure
        # Training sees: [prompt1][speech][prompt2_no_colon][colon][text]
        # Generation must start with: [prompt1][speech][prompt2_no_colon][colon]
        start_token_emb = self.emb_cast(
            torch.tensor([[self.last_token_id]], device=speech_emb.device)
        )
        start_token_emb = repeat(start_token_emb, "1 1 d -> b 1 d", b=B)
        z = torch.cat([prompt1, speech_emb, prompt2, start_token_emb], dim=1)

        self.p_c = z.shape[1]

        attention_mask = torch.ones(B, z.shape[1], dtype=torch.long, device=z.device)
        if self.num_of_beams == 1:
            generate_ids = self.model.generate(
                inputs_embeds=z,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.eos_token_id,
            )
        elif self.num_of_beams > 1:
            generate_ids = self.model.generate(
                inputs_embeds=z,
                attention_mask=attention_mask,
                max_new_tokens=128,
                num_beams=self.num_of_beams,
                do_sample=False,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.eos_token_id,
            )
        dec = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return dec

    def set_num_of_beams(self, num_of_beams: int):
        self.num_of_beams = num_of_beams


class LLMFactory:
    llms = {"qwen": Qwen, "gemma": Gemma}

    @classmethod
    def create(cls, model_id: str) -> LLM:
        if model_id not in cls.llms:
            return GenericLLM(model_id)
        return cls.llms[model_id](model_id)
    