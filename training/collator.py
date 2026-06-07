"""Data collators for multimodal chat SFT."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from training.image_refs import TessStage1ImageStore, load_image


class VLADataCollator:
    """Apply a processor chat template and build assistant-only labels."""

    def __init__(self, processor: Any, max_length: int = 4096, mask_prompt_labels: bool = True) -> None:
        self.processor = processor
        self.max_length = max_length
        self.mask_prompt_labels = mask_prompt_labels
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
