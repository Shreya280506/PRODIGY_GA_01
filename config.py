import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gpt2_text_generation")


def get_device():
    """Lazy load torch to avoid import errors if not installed."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("GPU detected: %s", torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            logger.warning("No GPU detected — falling back to CPU. Training will be slow.")
        return device
    except ImportError:
        logger.warning("PyTorch not installed. Using CPU as default device.")
        return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    logger.info("Random seed set to %d", seed)


@dataclass
class PathConfig:
    project_root: Path = Path(__file__).resolve().parent
    dataset_dir: Path = field(init=False)
    saved_model_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    processed_data_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.dataset_dir = self.project_root / "dataset"
        self.saved_model_dir = self.project_root / "saved_model"
        self.checkpoint_dir = self.project_root / "checkpoints"
        self.log_dir = self.project_root / "logs"
        self.processed_data_path = self.dataset_dir / "processed_data.txt"
        for directory in (self.saved_model_dir, self.checkpoint_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    raw_filename: str = "quotes.json"
    text_column: str = "Quote"
    min_char_length: int = 5
    train_split_ratio: float = 0.9


@dataclass
class ModelConfig:
    model_name: str = "gpt2"
    max_length: int = 128


@dataclass
class TrainingConfig:
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_epochs: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    save_every_n_steps: int = 200
    eval_every_n_steps: int = 200
    seed: int = 42
    fp16: bool = True


@dataclass
class GenerationConfig:
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    num_return_sequences: int = 1
    do_sample: bool = True


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    device: object = field(default_factory=get_device)

    def __post_init__(self) -> None:
        set_seed(self.training.seed)


CONFIG = Config()


if __name__ == "__main__":
    logger.info("Project root      : %s", CONFIG.paths.project_root)
    logger.info("Device             : %s", CONFIG.device)
    logger.info("Model              : %s", CONFIG.model.model_name)
    logger.info(
        "Batch size         : %d (effective: %d)",
        CONFIG.training.batch_size,
        CONFIG.training.batch_size * CONFIG.training.gradient_accumulation_steps,
    )
    logger.info("Saved model dir    : %s", CONFIG.paths.saved_model_dir)
