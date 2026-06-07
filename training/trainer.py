"""Trainer variants for Minecraft VLA SFT."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import Trainer


class ActionWeightedTrainer(Trainer):
    """Apply an action-region token weight on top of causal LM loss.

    The CombatVLA paper describes a richer adaptive action-weighted objective
    with language, action-alignment, and contrastive terms. The public training
    scaffold cannot reproduce its private action matcher exactly, but this keeps
    the most important available signal explicit: tokens inside normalized action
    JSON are weighted more heavily than reasoning tokens.
    """

    def __init__(
        self,
        *args: Any,
        action_token_weight: float = 1.0,
        action_start_token_ids: Optional[list[int]] = None,
        trunc_token_ids: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> None:
        self.sampler = kwargs.pop("sampler", "default")
        super().__init__(*args, **kwargs)
        self.action_token_weight = float(action_token_weight)
        self.action_start_token_ids = action_start_token_ids or []
        self.trunc_token_ids = trunc_token_ids or []

    def get_train_dataloader(self) -> DataLoader:
        if self.sampler != "sequential":
            return super().get_train_dataloader()
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if self.args.world_size > 1:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                shuffle=False,
            )
        else:
            sampler = SequentialSampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        if self.action_token_weight <= 1.0:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        labels = inputs.get("labels")
        if labels is None:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)

        weights = torch.ones_like(per_token_loss)
        label_mask = shift_labels.ne(-100)
        action_mask = self._action_region_mask(labels)[..., 1:].contiguous() & label_mask
        weights = torch.where(action_mask, weights * self.action_token_weight, weights)
        weights = torch.where(label_mask, weights, torch.zeros_like(weights))

        denominator = weights.sum().clamp_min(1.0)
        loss = (per_token_loss * weights).sum() / denominator
        return (loss, outputs) if return_outputs else loss

    def _action_region_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """Mark answer tokens from the action JSON start through `<TRUNC>`."""

        mask = torch.zeros_like(labels, dtype=torch.bool)
        for row in range(labels.shape[0]):
            token_ids = labels[row]
            valid_positions = torch.nonzero(token_ids.ne(-100), as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue

            start = self._find_subsequence(token_ids, self.action_start_token_ids, valid_positions)
            if start is None:
                start = int(valid_positions[0])

            end = self._find_subsequence(token_ids, self.trunc_token_ids, valid_positions, start=start)
            if end is None:
                end = int(valid_positions[-1]) + 1

            mask[row, start:end] = True

        return mask

    @staticmethod
    def _find_subsequence(
        token_ids: torch.Tensor,
        pattern: list[int],
        valid_positions: torch.Tensor,
        *,
        start: Optional[int] = None,
    ) -> Optional[int]:
        if not pattern:
            return None

        pattern_tensor = torch.tensor(pattern, dtype=token_ids.dtype, device=token_ids.device)
        min_pos = int(valid_positions[0]) if start is None else start
        max_pos = int(valid_positions[-1]) - len(pattern) + 2
        for pos in range(min_pos, max_pos):
            window = token_ids[pos : pos + len(pattern)]
            if window.shape[0] == len(pattern) and torch.equal(window, pattern_tensor):
                return pos
        return None
