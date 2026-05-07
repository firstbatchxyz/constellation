"""Masked full-SFT training entrypoint for curated pilot shards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from constellation.config import load_json_config
from constellation.formatting import IGNORE_INDEX, tokenize_with_loss_mask
from constellation.io import iter_jsonl
from constellation.schema import CanonicalSample


class OptionalTrainingDependencyError(RuntimeError):
    """Raised when GPU training dependencies are not installed."""


class JsonlSftDataset:
    def __init__(self, path: str | Path, tokenizer: Any, max_length: int) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = [CanonicalSample.from_dict(row) for row in iter_jsonl(self.path)]
        if not self.samples:
            raise ValueError(f"{self.path} contains no SFT samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return tokenize_with_loss_mask(
            self.tokenizer,
            self.samples[index],
            max_length=self.max_length,
        )


def require_training_dependencies() -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise OptionalTrainingDependencyError(
            "train-sft requires GPU training dependencies. Install them on the GPU machine with:\n"
            "uv pip install torch transformers accelerate trl datasets"
        ) from exc

    try:
        from trl import SFTTrainer
    except ImportError:
        SFTTrainer = None

    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "SFTTrainer": SFTTrainer,
    }


def ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token


def make_collator(tokenizer: Any) -> Any:
    import torch

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention_mask: list[list[int]] = []

        for feature in features:
            pad = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_token_id] * pad)
            labels.append(feature["labels"] + [IGNORE_INDEX] * pad)
            attention_mask.append(feature["attention_mask"] + [0] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    return collate


def build_trainer(
    *,
    deps: dict[str, Any],
    model: Any,
    args: Any,
    train_dataset: JsonlSftDataset,
    data_collator: Any,
    tokenizer: Any,
    use_trl_sft_trainer: bool,
) -> Any:
    sft_trainer = deps["SFTTrainer"]
    if use_trl_sft_trainer and sft_trainer is not None:
        try:
            return sft_trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                data_collator=data_collator,
                processing_class=tokenizer,
            )
        except TypeError:
            try:
                return sft_trainer(
                    model=model,
                    args=args,
                    train_dataset=train_dataset,
                    data_collator=data_collator,
                    tokenizer=tokenizer,
                )
            except TypeError:
                pass

    return deps["Trainer"](
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )


def train_sft_from_config(config_path: str | Path) -> dict[str, Any]:
    config = load_json_config(config_path)
    deps = require_training_dependencies()
    torch = deps["torch"]

    model_name = config["model_name_or_path"]
    output_dir = Path(config["output_dir"]).expanduser()
    train_file = Path(config["train_file"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = deps["AutoTokenizer"].from_pretrained(
        model_name,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        use_fast=bool(config.get("use_fast_tokenizer", True)),
    )
    ensure_pad_token(tokenizer)

    dtype = torch.bfloat16 if bool(config.get("bf16", True)) else None
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if config.get("attn_implementation"):
        model_kwargs["attn_implementation"] = config["attn_implementation"]

    model = deps["AutoModelForCausalLM"].from_pretrained(model_name, **model_kwargs)
    if bool(config.get("gradient_checkpointing", True)):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    train_dataset = JsonlSftDataset(
        train_file,
        tokenizer,
        max_length=int(config.get("max_seq_length", 8192)),
    )

    training_args = deps["TrainingArguments"](
        output_dir=str(output_dir),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 32)),
        learning_rate=float(config.get("learning_rate", 1e-5)),
        num_train_epochs=float(config.get("num_train_epochs", 1)),
        max_steps=int(config.get("max_steps", -1)),
        warmup_ratio=float(config.get("warmup_ratio", 0.03)),
        logging_steps=int(config.get("logging_steps", 10)),
        save_steps=int(config.get("save_steps", 200)),
        save_total_limit=int(config.get("save_total_limit", 2)),
        bf16=bool(config.get("bf16", True)),
        fp16=bool(config.get("fp16", False)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        remove_unused_columns=False,
        report_to=config.get("report_to", "none"),
    )

    trainer = build_trainer(
        deps=deps,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=make_collator(tokenizer),
        tokenizer=tokenizer,
        use_trl_sft_trainer=bool(config.get("use_trl_sft_trainer", False)),
    )
    result = trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    manifest = {
        "config_path": str(config_path),
        "model_name_or_path": model_name,
        "train_file": str(train_file),
        "output_dir": str(output_dir),
        "num_samples": len(train_dataset),
        "train_result": result.metrics,
    }
    (output_dir / "train_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest
