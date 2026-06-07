"""Data collators for multimodal chat SFT."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from training.actions import (
    MINECRAFT_ACTION_PRIORITY,
    highest_priority_action_category,
    infer_action_categories_from_actions,
    priority_alpha,
)
from training.image_refs import TessStage1ImageStore, load_image


class VLADataCollator:
    """Apply a processor chat template and build assistant-only labels."""

    def __init__(
        self,
        processor: Any,
        max_length: int = 4096,
        mask_prompt_labels: bool = True,
        action_priority: Optional[Sequence[str]] = None,
        action_alpha_min: float = 0.1,
        action_alpha_max: float = 1.0,
    ) -> None:
        self.processor = processor
        self.max_length = max_length
        self.mask_prompt_labels = mask_prompt_labels
        self.action_priority = list(action_priority or MINECRAFT_ACTION_PRIORITY)
        self.action_alpha_min = float(action_alpha_min)
        self.action_alpha_max = float(action_alpha_max)
        self._tess_stage1_stores: Dict[str, TessStage1ImageStore] = {}

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [
            self.processor.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            for item in examples
        ]

        images = []
        for item in examples:
            frame_images = []
            store = self._store_for(item.get("tess_stage1_index"))
            for frame in item["frames"]:
                frame_images.append(load_image(frame, store))
            images.append(frame_images)

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        if self.mask_prompt_labels:
            for row, item in enumerate(examples):
                prompt_messages = [msg for msg in item["messages"] if msg["role"] != "assistant"]
                prompt_text = self.processor.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_batch = self.processor(
                    text=[prompt_text],
                    images=[images[row]],
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                prompt_len = min(prompt_batch["input_ids"].shape[-1], labels.shape[-1])
                labels[row, :prompt_len] = -100

        batch["labels"] = labels
        category_ids = []
        alphas = []
        for item in examples:
            categories = infer_action_categories_from_actions(item.get("actions", []))
            category = highest_priority_action_category(categories, self.action_priority)
            try:
                category_id = self.action_priority.index(category)
            except ValueError:
                category_id = len(self.action_priority) - 1
            category_ids.append(category_id)
            alphas.append(
                priority_alpha(
                    category,
                    self.action_priority,
                    alpha_min=self.action_alpha_min,
                    alpha_max=self.action_alpha_max,
                )
            )
        batch["action_category_ids"] = torch.tensor(category_ids, dtype=torch.long)
        batch["action_priority_alpha"] = torch.tensor(alphas, dtype=torch.float)
        return batch

    def _store_for(self, index_path: Any) -> TessStage1ImageStore | None:
        if not index_path:
            return None
        key = str(index_path)
        store = self._tess_stage1_stores.get(key)
        if store is None:
            store = TessStage1ImageStore(key)
            self._tess_stage1_stores[key] = store
        return store
