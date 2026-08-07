import re
from pathlib import Path
from typing import List

import pandas as pd

from config import CONFIG, logger


class DataPreprocessor:
    EOS_TOKEN = "<|endoftext|>"
    SUPPORTED_LOADERS = {
        ".csv": pd.read_csv,
        ".json": pd.read_json,
    }

    def __init__(self, raw_data_path: Path, text_column: str, min_char_length: int = 5) -> None:
        self.raw_data_path = raw_data_path
        self.text_column = text_column
        self.min_char_length = min_char_length

    def load_raw_data(self) -> pd.DataFrame:
        if not self.raw_data_path.exists():
            raise FileNotFoundError(
                f"Could not find dataset at '{self.raw_data_path}'. Upload your dataset into the dataset/ folder first."
            )

        suffix = self.raw_data_path.suffix.lower()
        if suffix not in self.SUPPORTED_LOADERS:
            raise ValueError(
                f"Unsupported file type '{suffix}'. Supported types: {list(self.SUPPORTED_LOADERS.keys())}"
            )

        reader_fn = self.SUPPORTED_LOADERS[suffix]
        df = reader_fn(self.raw_data_path)
        logger.info("Loaded %d rows from %s", len(df), self.raw_data_path.name)

        if self.text_column not in df.columns:
            raise ValueError(
                f"Column '{self.text_column}' not found in {self.raw_data_path.name}. Available columns: {list(df.columns)}"
            )

        return df

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract_and_clean(self, df: pd.DataFrame) -> List[str]:
        series = df[self.text_column].dropna()
        cleaned = [self.clean_text(str(quote)) for quote in series]
        cleaned = [q for q in cleaned if len(q) >= self.min_char_length]
        after_length_filter = len(cleaned)
        cleaned = list(dict.fromkeys(cleaned))
        duplicates_removed = after_length_filter - len(cleaned)
        missing_or_short = len(df) - after_length_filter
        logger.info(
            "Kept %d quotes | discarded %d (missing/empty/too short) | removed %d exact duplicates",
            len(cleaned),
            missing_or_short,
            duplicates_removed,
        )
        return cleaned

    def add_boundary_tokens(self, quotes: List[str]) -> List[str]:
        return [f"{quote} {self.EOS_TOKEN}" for quote in quotes]

    def save(self, quotes: List[str], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(quotes))
        logger.info("Saved %d processed quotes to %s", len(quotes), output_path)

    def run(self, output_path: Path) -> List[str]:
        df = self.load_raw_data()
        quotes = self.extract_and_clean(df)
        quotes = self.add_boundary_tokens(quotes)
        self.save(quotes, output_path)
        return quotes


if __name__ == "__main__":
    raw_path = CONFIG.paths.dataset_dir / CONFIG.data.raw_filename
    preprocessor = DataPreprocessor(
        raw_data_path=raw_path,
        text_column=CONFIG.data.text_column,
        min_char_length=CONFIG.data.min_char_length,
    )
    try:
        preprocessor.run(output_path=CONFIG.paths.processed_data_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
