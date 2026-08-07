from pathlib import Path
from typing import Tuple

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, PreTrainedTokenizer

from config import CONFIG, logger


class GPT2ModelManager:
    def __init__(self, model_name: str, device: torch.device) -> None:
        self.model_name = model_name
        self.device = device

    def load_pretrained(self) -> GPT2LMHeadModel:
        model = GPT2LMHeadModel.from_pretrained(self.model_name)
        model.to(self.device)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info("Loaded '%s' (%d params) on %s", self.model_name, num_params, self.device)
        return model

    @staticmethod
    def save(model: GPT2LMHeadModel, tokenizer: PreTrainedTokenizer, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Saved model + tokenizer to %s", output_dir)

    def load_finetuned(self, model_dir: Path) -> Tuple[GPT2LMHeadModel, PreTrainedTokenizer]:
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        model.to(self.device)
        logger.info("Loaded fine-tuned model from %s", model_dir)
        return model, tokenizer


if __name__ == "__main__":
    manager = GPT2ModelManager(CONFIG.model.model_name, CONFIG.device)
    model = manager.load_pretrained()
    logger.info("n_layer=%d n_head=%d n_embd=%d", model.config.n_layer, model.config.n_head, model.config.n_embd)
