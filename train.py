import math
from typing import Dict, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from config import CONFIG, Config, logger
from dataset import build_datasets, load_tokenizer
from model import GPT2ModelManager


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = math.inf
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {key: torch.stack([example[key] for example in batch]) for key in batch[0].keys()}


class Trainer:
    def __init__(self, model, tokenizer, train_loader, val_loader, device, cfg: Config) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        total_steps = (len(train_loader) // cfg.training.gradient_accumulation_steps) * cfg.training.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=cfg.training.warmup_steps,
            num_training_steps=max(total_steps, 1),
        )

        self.amp_enabled = cfg.training.fp16 and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.writer = SummaryWriter(log_dir=str(cfg.paths.log_dir))
        self.early_stopping = EarlyStopping(patience=cfg.training.early_stopping_patience)
        self.global_step = 0

    def _optimizer_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1

    def _log_and_checkpoint(self, loss_value: float) -> bool:
        self.writer.add_scalar("train/loss", loss_value, self.global_step)
        self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)

        if self.global_step % self.cfg.training.save_every_n_steps == 0:
            self.save_checkpoint(f"checkpoint-step-{self.global_step}")

        if self.global_step % self.cfg.training.eval_every_n_steps == 0:
            val_loss = self.evaluate()
            self.writer.add_scalar("val/loss", val_loss, self.global_step)
            self.writer.add_scalar("val/perplexity", math.exp(val_loss), self.global_step)
            logger.info("step=%d val_loss=%.4f val_ppl=%.2f", self.global_step, val_loss, math.exp(val_loss))
            self.model.train()
            return self.early_stopping.step(val_loss)

        return False

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        accum_steps = self.cfg.training.gradient_accumulation_steps
        self.optimizer.zero_grad()

        progress = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accum_steps

            self.scaler.scale(loss).backward()
            batch_loss = loss.item() * accum_steps
            total_loss += batch_loss
            progress.set_postfix(loss=batch_loss)

            is_last_batch = (step + 1) == num_batches
            if (step + 1) % accum_steps == 0 or is_last_batch:
                self._optimizer_step()
                if self._log_and_checkpoint(batch_loss):
                    return total_loss / (step + 1)

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
        return total_loss / len(self.val_loader)

    def save_checkpoint(self, name: str) -> None:
        path = self.cfg.paths.checkpoint_dir / name
        GPT2ModelManager.save(self.model, self.tokenizer, path)

    def fit(self) -> None:
        for epoch in range(1, self.cfg.training.num_epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            logger.info("epoch=%d avg_train_loss=%.4f", epoch, train_loss)
            if self.early_stopping.should_stop:
                logger.info("Early stopping after epoch %d", epoch)
                break

        GPT2ModelManager.save(self.model, self.tokenizer, self.cfg.paths.saved_model_dir)
        self.writer.close()
        logger.info("Training complete. Model saved to %s", self.cfg.paths.saved_model_dir)


if __name__ == "__main__":
    tokenizer = load_tokenizer(CONFIG.model.model_name)
    train_ds, val_ds, max_length = build_datasets(
        tokenizer=tokenizer,
        file_path=CONFIG.paths.processed_data_path,
        hard_cap_length=CONFIG.model.max_length,
        train_split_ratio=CONFIG.data.train_split_ratio,
        seed=CONFIG.training.seed,
    )

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG.training.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG.training.batch_size, shuffle=False, collate_fn=collate_fn
    )

    model_manager = GPT2ModelManager(CONFIG.model.model_name, CONFIG.device)
    model = model_manager.load_pretrained()

    trainer = Trainer(model, tokenizer, train_loader, val_loader, CONFIG.device, CONFIG)
    trainer.fit()
