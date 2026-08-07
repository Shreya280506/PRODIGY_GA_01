from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer, PreTrainedTokenizer

from config import CONFIG, logger


def load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    logger.info("Tokenizer loaded. Vocabulary size: %d", tokenizer.vocab_size)
    return tokenizer


def load_processed_lines(file_path: Path) -> List[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"'{file_path}' not found. Run preprocess.py first to generate it.")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    logger.info("Loaded %d processed lines from %s", len(lines), file_path.name)
    return lines


def compute_max_length(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    percentile: float = 95.0,
    hard_cap: int = 128,
) -> int:
    lengths = [len(tokenizer.encode(t)) for t in tqdm(texts, desc="Measuring token lengths")]
    lengths_arr = np.array(lengths)
    recommended = int(np.percentile(lengths_arr, percentile))
    recommended = min(((recommended + 7) // 8) * 8, hard_cap)
    logger.info(
        "Token lengths — mean: %.1f, median: %d, p95-based max_length: %d, true max: %d",
        lengths_arr.mean(),
        int(np.median(lengths_arr)),
        recommended,
        int(lengths_arr.max()),
    )
    return recommended


class QuoteDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: PreTrainedTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[Dict[str, torch.Tensor]] = []

        for text in tqdm(texts, desc="Tokenizing"):
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


def build_datasets(
    tokenizer: PreTrainedTokenizer,
    file_path: Path,
    hard_cap_length: int,
    train_split_ratio: float,
    seed: int,
) -> Tuple[QuoteDataset, QuoteDataset, int]:
    lines = load_processed_lines(file_path)
    max_length = compute_max_length(lines, tokenizer, hard_cap=hard_cap_length)
    train_lines, val_lines = train_test_split(lines, train_size=train_split_ratio, random_state=seed, shuffle=True)
    logger.info("Split: %d train / %d validation", len(train_lines), len(val_lines))
    train_dataset = QuoteDataset(train_lines, tokenizer, max_length)
    val_dataset = QuoteDataset(val_lines, tokenizer, max_length)
    return train_dataset, val_dataset, max_length


if __name__ == "__main__":
    tokenizer = load_tokenizer(CONFIG.model.model_name)
    train_ds, val_ds, max_length = build_datasets(
        tokenizer=tokenizer,
        file_path=CONFIG.paths.processed_data_path,
        hard_cap_length=CONFIG.model.max_length,
        train_split_ratio=CONFIG.data.train_split_ratio,
        seed=CONFIG.training.seed,
    )
    logger.info("Train size: %d | Val size: %d | max_length: %d", len(train_ds), len(val_ds), max_length)
    sample = train_ds[0]
    logger.info("Sample input_ids shape: %s", sample["input_ids"].shape)
    logger.info("Sample decoded: %s", tokenizer.decode(sample["input_ids"], skip_special_tokens=True))
