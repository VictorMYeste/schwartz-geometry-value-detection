"""Transformers-backed chat model client for the LLM diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

from schwartz_value_geometry.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class LLMClientConfig:
    model_name: str = "Qwen/Qwen2.5-72B-Instruct"
    device: str | None = None
    quantization: str | None = "4bit"
    int8_fp32_cpu_offload: bool = False
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    trust_remote_code: bool = False


class TransformersChatClient:
    """Minimal chat-template-aware generation client."""

    def __init__(self, config: LLMClientConfig) -> None:
        self.config = config
        self.processor: Any | None = None
        self._use_chat_template = False
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import (  # type: ignore
                AutoModelForCausalLM,
                AutoProcessor,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("transformers and torch are required for LLM inference") from exc

        self.torch = torch
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Loading LLM model %s", self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
            )
            processor_tokenizer = getattr(self.processor, "tokenizer", None)
            if processor_tokenizer is not None:
                self.tokenizer = processor_tokenizer
            self._use_chat_template = hasattr(self.processor, "apply_chat_template")
        except Exception:
            self.processor = None
            self._use_chat_template = (
                hasattr(self.tokenizer, "apply_chat_template")
                and getattr(self.tokenizer, "chat_template", None) is not None
            )

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.bfloat16
        quantization = (self.config.quantization or "none").strip().lower()
        if quantization == "8bit":
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"MatMul8bitLt: inputs will be cast from torch\.bfloat16 "
                    r"to float16 during quantization"
                ),
                category=UserWarning,
            )
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=self.config.int8_fp32_cpu_offload,
            )
        elif quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization not in {"none", ""}:
            raise ValueError(f"Unsupported quantization mode: {self.config.quantization}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        self.model.eval()
        if self.device != "cuda":
            self.model.to(self.device)

    @staticmethod
    def _messages(prompt: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    def preview_model_prompt(self, prompt: str) -> str:
        """Return the prompt as rendered by the tokenizer chat template."""
        if not self._use_chat_template:
            return prompt
        messages = self._messages(prompt)
        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            try:
                return str(
                    self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                )
            except Exception:
                pass
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return str(
                    self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                )
            except Exception:
                pass
        return prompt

    def _encode(self, prompt: str):
        if self._use_chat_template:
            messages = self._messages(prompt)
            if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
                try:
                    return self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                except Exception:
                    pass
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        return self.tokenizer(prompt, return_tensors="pt")

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        torch = self.torch
        max_tokens = max_tokens or self.config.max_new_tokens
        temperature = self.config.temperature if temperature is None else temperature
        top_p = self.config.top_p if top_p is None else top_p
        inputs = self._encode(prompt)
        if isinstance(inputs, torch.Tensor):
            inputs = {"input_ids": inputs}
        if "attention_mask" not in inputs and "input_ids" in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        first_device = next(self.model.parameters()).device
        inputs = {key: value.to(first_device) for key, value in inputs.items()}

        eos_ids: list[int] = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(int(self.tokenizer.eos_token_id))
        vocab = self.tokenizer.get_vocab() if hasattr(self.tokenizer, "get_vocab") else {}
        for token in ("<|im_end|>", "<end_of_turn>"):
            if token in vocab and int(vocab[token]) not in eos_ids:
                eos_ids.append(int(vocab[token]))

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_tokens),
            "do_sample": float(temperature) > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }
        if eos_ids:
            generation_kwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]
        if float(temperature) > 0.0:
            generation_kwargs["temperature"] = float(temperature)
            generation_kwargs["top_p"] = float(top_p)

        with torch.no_grad():
            output = self.model.generate(**inputs, **generation_kwargs)
        generated = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
