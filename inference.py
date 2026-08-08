"""
GPT-2 Text Generation with multiple decoding strategies.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from config import CONFIG, logger


@dataclass
class GenerationParams:
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    num_return_sequences: int = 1
    num_beams: int = 1
    do_sample: bool = True


class TextGenerator:

    def __init__(self, model_dir: Path, device) -> None:
        try:
            import torch
            from transformers import GPT2LMHeadModel, AutoTokenizer

            self.device = device

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

            # GPT-2 needs a padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load your trained model
            self.model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
            self.model.eval()

            logger.info("TextGenerator ready (device=%s)", device)

        except ImportError as e:
            raise ImportError(
                f"PyTorch and transformers required for TextGenerator: {e}"
            ) from e

    def generate(self, prompt: str, params: GenerationParams) -> List[str]:
        import torch

        with torch.no_grad():

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self.device)

            generation_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": params.max_new_tokens,
                "repetition_penalty": params.repetition_penalty,
                "num_return_sequences": params.num_return_sequences,
                "num_beams": params.num_beams,
                "do_sample": params.do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            # Sampling parameters are only used when sampling is enabled
            if params.do_sample:
                generation_kwargs.update({
                    "temperature": params.temperature,
                    "top_k": params.top_k,
                    "top_p": params.top_p,
                })

            output_ids = self.model.generate(**generation_kwargs)

            return [
                self.tokenizer.decode(
                    ids,
                    skip_special_tokens=True
                )
                for ids in output_ids
            ]

    def greedy(self, prompt: str, max_new_tokens: int = 100) -> str:
        params = GenerationParams(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1
        )

        return self.generate(prompt, params)[0]

    def beam_search(
        self,
        prompt: str,
        num_beams: int = 5,
        max_new_tokens: int = 100
    ) -> str:

        params = GenerationParams(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams
        )

        return self.generate(prompt, params)[0]

    def top_k_sampling(
        self,
        prompt: str,
        top_k: int = 50,
        max_new_tokens: int = 100
    ) -> str:

        params = GenerationParams(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=top_k,
            top_p=1.0
        )

        return self.generate(prompt, params)[0]

    def top_p_sampling(
        self,
        prompt: str,
        top_p: float = 0.9,
        max_new_tokens: int = 100
    ) -> str:

        params = GenerationParams(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=0,
            top_p=top_p
        )

        return self.generate(prompt, params)[0]

    def temperature_sampling(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_new_tokens: int = 100
    ) -> str:

        params = GenerationParams(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=0,
            top_p=1.0
        )

        return self.generate(prompt, params)[0]


if __name__ == "__main__":

    generator = TextGenerator(
        CONFIG.paths.saved_model_dir,
        CONFIG.device
    )

    print("\n" + "=" * 60)
    print("        GPT-2 TEXT GENERATION")
    print("=" * 60)

    prompt = input("\nEnter your prompt: ").strip()

    if not prompt:
        prompt = "Life is"

    print("\nGenerating text...\n")

    print("GREEDY:")
    print(generator.greedy(prompt, max_new_tokens=40))

    print("\nBEAM SEARCH:")
    print(generator.beam_search(
        prompt,
        num_beams=5,
        max_new_tokens=40
    ))

    print("\nTOP-K SAMPLING:")
    print(generator.top_k_sampling(
        prompt,
        top_k=50,
        max_new_tokens=40
    ))

    print("\nTOP-P SAMPLING:")
    print(generator.top_p_sampling(
        prompt,
        top_p=0.9,
        max_new_tokens=40
    ))

    print("\nTEMPERATURE SAMPLING:")
    print(generator.temperature_sampling(
        prompt,
        temperature=1.2,
        max_new_tokens=40
    ))

    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)