"""SFT entrypoint for Minecraft VLA data."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import Subset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from training.collator import VLADataCollator
from training.dataset import MinecraftVLADataset, manifest_record_count
from training.aot import TRUNC_TOKEN
from training.actions import ACTION_CATEGORY_ALIASES, MINECRAFT_ACTION_PRIORITY
from training.trainer import ActionWeightedTrainer


DEFAULT_VISION_PREFIXES = ["visual", "vision_tower", "vision_model", "visual_model", "vision_encoder"]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def parse_override_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def apply_override(config: Dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must be KEY=VALUE, got: {override}")
    dotted_key, value = override.split("=", 1)
    keys = [part for part in dotted_key.split(".") if part]
    if not keys:
        raise ValueError(f"Invalid override key: {dotted_key!r}")
    target: Dict[str, Any] = config
    for key in keys[:-1]:
        child = target.get(key)
        if child is None:
            child = {}
            target[key] = child
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {dotted_key}: {key} is not an object.")
        target = child
    target[keys[-1]] = parse_override_value(value)


def optional_quantization(config: Dict[str, Any]) -> Optional[BitsAndBytesConfig]:
    quant = config.get("quantization") or {}
    if not quant.get("load_in_4bit", False):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(torch, quant.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=quant.get("bnb_4bit_use_double_quant", True),
    )


def set_trainability(model: torch.nn.Module, config: Dict[str, Any]) -> None:
    freeze_prefixes = config.get("freeze_parameter_prefixes")
    if freeze_prefixes is None and config.get("freeze_vision_encoder", True):
        freeze_prefixes = DEFAULT_VISION_PREFIXES
    freeze_prefixes = freeze_prefixes or []

    for name, parameter in model.named_parameters():
        if any(
            name == prefix
            or name.startswith(prefix + ".")
            or name.startswith("model." + prefix + ".")
            for prefix in freeze_prefixes
        ):
            parameter.requires_grad = False


def print_trainable_parameters(model: torch.nn.Module) -> None:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    ratio = 100 * trainable / total if total else 0
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {ratio:.4f}")


def build_model(config: Dict[str, Any]):
    model_name = config["model_name_or_path"]
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=getattr(torch, config.get("torch_dtype", "bfloat16")),
        device_map=config.get("device_map", None),
        quantization_config=optional_quantization(config),
    )

    extra_tokens = config.get("additional_special_tokens")
    if extra_tokens is None:
        extra_tokens = [TRUNC_TOKEN]
    if extra_tokens and hasattr(processor, "tokenizer"):
        added = processor.tokenizer.add_special_tokens({"additional_special_tokens": list(extra_tokens)})
        if added:
            model.resize_token_embeddings(len(processor.tokenizer))

    if config.get("gradient_checkpointing", True):
        if hasattr(model, "config"):
            model.config.use_cache = False
        model.gradient_checkpointing_enable()

    set_trainability(model, config)

    lora_cfg = config.get("lora")
    if lora_cfg:
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_cfg.get("r", 16),
                lora_alpha=lora_cfg.get("lora_alpha", 32),
                lora_dropout=lora_cfg.get("lora_dropout", 0.05),
                target_modules=lora_cfg.get(
                    "target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                ),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        model.print_trainable_parameters()
    else:
        print_trainable_parameters(model)

    return model, processor


def tokenizer_patterns_for_action_priority(tokenizer: Any, priority: list[str]) -> list[list[list[int]]]:
    patterns: list[list[list[int]]] = []
    for category in priority:
        aliases = ACTION_CATEGORY_ALIASES.get(category, [category])
        category_patterns: list[list[int]] = []
        for alias in aliases:
            raw_aliases = {str(alias), str(alias).lower()}
            text_variants = set()
            for raw_alias in raw_aliases:
                text_variants.update(
                    {
                        raw_alias,
                        f" {raw_alias}",
                        f'"{raw_alias}"',
                        f':"{raw_alias}"',
                    }
                )
            for text in text_variants:
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                if token_ids and token_ids not in category_patterns:
                    category_patterns.append(token_ids)
        patterns.append(category_patterns)
    return patterns


def image_token_ids(processor: Any) -> list[int]:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return []

    candidates = [
        "<image>",
        "<|image_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|video_pad|>",
    ]
    ids: list[int] = []
    for token in candidates:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            continue
        if isinstance(token_id, int) and token_id >= 0 and token_id not in ids:
            ids.append(token_id)
    return ids


def build_dataset(data_cfg: Dict[str, Any], manifest_key: str) -> MinecraftVLADataset:
    return MinecraftVLADataset(
        manifest_path=data_cfg[manifest_key],
        image_root=data_cfg.get("image_root"),
        stage=data_cfg.get("stage", 3),
        max_frames=data_cfg.get("max_frames", 3),
        system_prompt=data_cfg.get("system_prompt"),
        user_prompt=data_cfg.get("user_prompt"),
        tess_stage1_index=data_cfg.get("tess_stage1_index"),
        require_reasoning=bool(data_cfg.get("require_reasoning", False)),
    )


def preflight_dataset(dataset: torch.utils.data.Dataset, name: str) -> None:
    if len(dataset) == 0:
        raise ValueError(f"{name} dataset is empty.")
    try:
        dataset[0]
    except Exception as exc:
        raise ValueError(f"{name} dataset preflight failed before model loading.") from exc


def require_full_manifest_coverage(data_cfg: Dict[str, Any], split: str, dataset: torch.utils.data.Dataset) -> None:
    source_key = f"source_{split}_manifest"
    source_manifest = data_cfg.get(source_key)
    if not source_manifest:
        return
    expected = manifest_record_count(source_manifest)
    actual = len(dataset)
    if actual != expected:
        raise ValueError(
            f"{split} reasoning manifest is partial: {actual:,} records, "
            f"but {source_manifest} has {expected:,}. "
            "Paper-style full-data reproduction requires reasoning coverage for every source record."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Minecraft VLA model with CombatVLA-style AoT SFT.")
    parser.add_argument("--config", required=True, help="Path to JSON training config.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a config value, e.g. --set training.deepspeed=configs/deepspeed/zero2.json",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    for override in args.set:
        apply_override(config, override)

    data_cfg = config["data"]
    train_dataset = build_dataset(data_cfg, "train_manifest")
    eval_dataset = None
    if data_cfg.get("eval_manifest"):
        eval_dataset = build_dataset(data_cfg, "eval_manifest")
        eval_max_samples = int(data_cfg.get("eval_max_samples") or 0)
        if eval_max_samples > 0 and eval_max_samples < len(eval_dataset):
            eval_dataset = Subset(eval_dataset, range(eval_max_samples))

    preflight_dataset(train_dataset, "train")
    if eval_dataset is not None:
        preflight_dataset(eval_dataset, "eval")
    if data_cfg.get("require_full_coverage", False):
        require_full_manifest_coverage(data_cfg, "train", train_dataset)
        if eval_dataset is not None:
            require_full_manifest_coverage(data_cfg, "eval", eval_dataset)

    model, processor = build_model(config)

    train_cfg = config["training"]
    output_dir = Path(train_cfg.get("output_dir", "outputs/minecraft-vla"))
    output_dir.mkdir(parents=True, exist_ok=True)

    args_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": train_cfg.get("per_device_train_batch_size", 1),
        "per_device_eval_batch_size": train_cfg.get("per_device_eval_batch_size", 1),
        "gradient_accumulation_steps": train_cfg.get("gradient_accumulation_steps", 8),
        "learning_rate": train_cfg.get("learning_rate", 1e-5),
        "num_train_epochs": train_cfg.get("num_train_epochs", 1),
        "max_steps": train_cfg.get("max_steps", -1),
        "warmup_ratio": train_cfg.get("warmup_ratio", 0.03),
        "logging_steps": train_cfg.get("logging_steps", 10),
        "save_steps": train_cfg.get("save_steps", 500),
        "eval_steps": train_cfg.get("eval_steps", 500),
        "save_strategy": train_cfg.get("save_strategy", "steps"),
        "save_total_limit": train_cfg.get("save_total_limit", 2),
        "bf16": train_cfg.get("bf16", True),
        "fp16": train_cfg.get("fp16", False),
        "dataloader_num_workers": train_cfg.get("dataloader_num_workers", 0),
        "remove_unused_columns": False,
        "report_to": train_cfg.get("report_to", "none"),
    }
    optional_training_keys = [
        "deepspeed",
        "ddp_find_unused_parameters",
        "gradient_checkpointing",
        "gradient_checkpointing_kwargs",
        "max_grad_norm",
        "optim",
        "lr_scheduler_type",
    ]
    signature = inspect.signature(TrainingArguments).parameters
    for key in optional_training_keys:
        if key in train_cfg and key in signature:
            args_kwargs[key] = train_cfg[key]
    eval_key = "eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"
    args_kwargs[eval_key] = "steps" if eval_dataset is not None else "no"
    training_args = TrainingArguments(**args_kwargs)
    loss_cfg = config.get("loss", {})
    action_token_weight = float(loss_cfg.get("action_token_weight", 1.0))
    action_priority = list(loss_cfg.get("action_priority") or MINECRAFT_ACTION_PRIORITY)
    tokenizer = processor.tokenizer
    action_start_token_ids = tokenizer.encode('{"actions"', add_special_tokens=False)
    trunc_token_ids = tokenizer.encode(TRUNC_TOKEN, add_special_tokens=False)
    action_category_token_patterns = tokenizer_patterns_for_action_priority(tokenizer, action_priority)

    trainer = ActionWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=VLADataCollator(
            processor,
            max_length=data_cfg.get("max_length", 4096),
            mask_prompt_labels=data_cfg.get("mask_prompt_labels", True),
            action_priority=action_priority,
            action_alpha_min=float(loss_cfg.get("alpha_min", 0.1)),
            action_alpha_max=float(loss_cfg.get("alpha_max", 1.0)),
        ),
        action_token_weight=action_token_weight,
        action_start_token_ids=action_start_token_ids,
        trunc_token_ids=trunc_token_ids,
        action_category_token_patterns=action_category_token_patterns,
        action_priority=action_priority,
        decode_tokenizer=tokenizer,
        image_token_ids=image_token_ids(processor),
        adaptive_action_loss=bool(loss_cfg.get("adaptive_action_loss", True)),
        contrastive_weight=float(loss_cfg.get("contrastive_weight", 1.0)),
        alignment_weight=float(loss_cfg.get("alignment_weight", 1.0)),
        sampler=train_cfg.get("sampler", "default"),
    )

    resume_from_checkpoint = train_cfg.get("resume_from_checkpoint")
    if resume_from_checkpoint == "auto":
        resume_from_checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
        if resume_from_checkpoint:
            print(f"Resuming from latest checkpoint: {resume_from_checkpoint}", flush=True)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if train_cfg.get("save_final", True):
        trainer.save_model(str(output_dir / "final"))
        processor.save_pretrained(str(output_dir / "final"))


if __name__ == "__main__":
    main()
