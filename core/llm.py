"""
LLM Wrapper

Simple wrapper for local LLMs using transformers.
Provides a consistent interface for agents.
"""

import torch
from typing import Optional


class LocalLLM:
    """
    Local LLM wrapper using HuggingFace transformers.

    Supports Qwen, Llama, Mistral, and other instruction-tuned models.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = None,
        max_memory: dict = None,
        torch_dtype: str = "auto",
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading {model_name}...")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Load model with appropriate settings
        dtype = torch.bfloat16 if torch_dtype == "auto" else getattr(torch, torch_dtype)

        if self.device == "cuda" and torch.cuda.is_available():
            # Load directly to GPU to avoid meta device issues
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).to("cuda")
        else:
            # CPU loading with device_map
            load_kwargs = {
                "device_map": "auto" if max_memory else None,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            if max_memory:
                load_kwargs["max_memory"] = max_memory

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                **load_kwargs
            )
            if not max_memory and self.device != "cpu":
                self.model = self.model.to(self.device)

        print(f"Model loaded on {self.device}")

    def generate(self, prompt: str, max_tokens: int = 1000,
                 temperature: float = 0.7, stop: list[str] = None) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences

        Returns:
            Generated text (without prompt)
        """
        # Format as chat if model supports it
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted = prompt

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Apply stop sequences
        if stop:
            for s in stop:
                if s in response:
                    response = response[:response.index(s)]

        return response.strip()

    def complete(self, prompt: str, max_tokens: int = 1000) -> str:
        """Alias for generate (for API compatibility)."""
        return self.generate(prompt, max_tokens=max_tokens)


def load_qwen(size: str = "7B") -> LocalLLM:
    """
    Load a Qwen model.

    Args:
        size: Model size - "0.5B", "1.5B", "7B", "14B", or "coder-14B"

    Returns:
        LocalLLM instance
    """
    models = {
        "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
        "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
        "3B": "Qwen/Qwen2.5-3B-Instruct",
        "7B": "Qwen/Qwen2.5-7B-Instruct",
        "14B": "Qwen/Qwen2.5-14B-Instruct",
        "coder-1.5B": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "coder-3B": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "coder-7B": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "coder-14B": "Qwen/Qwen2.5-Coder-14B-Instruct",
    }

    model_name = models.get(size, models["7B"])
    return LocalLLM(model_name)


def load_deepseek(variant: str = "coder-6.7b") -> LocalLLM:
    """
    Load a DeepSeek model.

    Args:
        variant: Model variant:
            - "coder-1.3b": Fast, lightweight code model (~3GB VRAM)
            - "coder-6.7b": Balanced code model (~13GB VRAM)
            - "r1-7b": Reasoning-focused (R1-Distill-Qwen-7B)
            - "r1-1.5b": Small reasoning model

    Returns:
        LocalLLM instance
    """
    models = {
        "coder-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "coder-6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "r1-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "r1-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    }

    model_name = models.get(variant, models["coder-6.7b"])
    return LocalLLM(model_name)
