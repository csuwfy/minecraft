"""Trainer variants for Minecraft VLA SFT."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import Trainer

from training.actions import MINECRAFT_ACTION_PRIORITY, infer_action_categories_from_text


class ActionWeightedTrainer(Trainer):
    """CombatVLA-style adaptive action-weighted objective.

    The paper defines L = L_lang + alpha * L_act, where L_act uses a
    priority-aware match between predicted and gold action categories. Matched
    actions pull visual/action EOS embeddings together; mismatches push them
    apart and add an action-alignment CE term for the gold priority action.
    """

    def __init__(
        self,
        *args: Any,
        action_token_weight: float = 1.0,
        action_start_token_ids: Optional[list[int]] = None,
        trunc_token_ids: Optional[list[int]] = None,
        action_category_token_patterns: Optional[Sequence[Sequence[Sequence[int]]]] = None,
        action_priority: Optional[Sequence[str]] = None,
        image_token_ids: Optional[Sequence[int]] = None,
        adaptive_action_loss: bool = True,
        contrastive_weight: float = 1.0,
        alignment_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.sampler = kwargs.pop("sampler", "default")
        super().__init__(*args, **kwargs)
        self.action_token_weight = float(action_token_weight)
        self.action_start_token_ids = action_start_token_ids or []
        self.trunc_token_ids = trunc_token_ids or []
        self.action_category_token_patterns = [
            [list(pattern) for pattern in category_patterns if pattern]
            for category_patterns in (action_category_token_patterns or [])
        ]
        self.action_priority = list(action_priority or MINECRAFT_ACTION_PRIORITY)
        self.image_token_ids = list(image_token_ids or [])
        self.adaptive_action_loss = bool(adaptive_action_loss)
        self.contrastive_weight = float(contrastive_weight)
        self.alignment_weight = float(alignment_weight)

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
        action_category_ids = inputs.pop("action_category_ids", None)
        action_priority_alpha = inputs.pop("action_priority_alpha", None)

        if self.action_token_weight <= 1.0 and not self.adaptive_action_loss:
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

        outputs = model(**inputs, output_hidden_states=self.adaptive_action_loss)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)

        label_mask = shift_labels.ne(-100)
        weights = torch.ones_like(per_token_loss)
        if self.action_token_weight > 1.0:
            action_mask = self._action_region_mask(labels)[..., 1:].contiguous() & label_mask
            weights = torch.where(action_mask, weights * self.action_token_weight, weights)
        weights = torch.where(label_mask, weights, torch.zeros_like(weights))

        denominator = weights.sum().clamp_min(1.0)
        language_loss = (per_token_loss * weights).sum() / denominator
        loss = language_loss

        if self.adaptive_action_loss:
            action_loss = self._adaptive_action_loss(
                outputs=outputs,
                logits=logits,
                labels=labels,
                input_ids=inputs.get("input_ids"),
                action_category_ids=action_category_ids,
                action_priority_alpha=action_priority_alpha,
            )
            loss = language_loss + action_loss

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

    def _adaptive_action_loss(
        self,
        *,
        outputs: Any,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        action_category_ids: Optional[torch.Tensor],
        action_priority_alpha: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if action_category_ids is None or action_priority_alpha is None:
            return logits.new_zeros(())
        if not self.action_category_token_patterns:
            return logits.new_zeros(())
        if not getattr(outputs, "hidden_states", None):
            return logits.new_zeros(())

        hidden = outputs.hidden_states[-1]
        batch_losses = []
        for row in range(labels.shape[0]):
            valid_positions = torch.nonzero(labels[row].ne(-100), as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue

            gold_id = int(action_category_ids[row].detach().cpu().item())
            if gold_id < 0 or gold_id >= len(self.action_category_token_patterns):
                continue

            predicted_ids = torch.full_like(labels[row], fill_value=-100)
            if predicted_ids.numel() > 1:
                predicted_ids[1:] = logits[row, :-1, :].argmax(dim=-1)
            action_positions = torch.nonzero(self._action_region_mask(labels[row : row + 1])[0], as_tuple=False).flatten()
            predicted_match = self._predicted_category_matches(
                predicted_ids,
                gold_id,
                self.action_category_token_patterns[gold_id],
                action_positions,
            )

            visual_position = self._visual_eos_position(input_ids[row] if input_ids is not None else None, valid_positions)
            action_position = int(valid_positions[-1])
            visual_repr = hidden[row, visual_position, :]
            action_repr = hidden[row, action_position, :]
            pull_contrastive = 1.0 - F.cosine_similarity(
                visual_repr.float(),
                action_repr.float(),
                dim=0,
                eps=1e-8,
            )

            if predicted_match:
                action_loss = self.contrastive_weight * pull_contrastive
            else:
                alignment_loss = self._alignment_loss_for_category(
                    logits[row],
                    labels[row],
                    self.action_category_token_patterns[gold_id],
                )
                action_loss = (
                    self.contrastive_weight * (-pull_contrastive)
                    + self.alignment_weight * alignment_loss
                )

            alpha = action_priority_alpha[row].to(device=logits.device, dtype=logits.dtype)
            batch_losses.append(alpha * action_loss)

        if not batch_losses:
            return logits.new_zeros(())
        return torch.stack(batch_losses).mean()

    def _visual_eos_position(self, input_ids: Optional[torch.Tensor], valid_positions: torch.Tensor) -> int:
        if input_ids is None or not self.image_token_ids:
            return int(valid_positions[0])
        image_token_set = set(self.image_token_ids)
        image_positions = [
            int(pos)
            for pos in torch.nonzero(input_ids.ne(-100), as_tuple=False).flatten().detach().cpu().tolist()
            if int(input_ids[pos].detach().cpu().item()) in image_token_set
        ]
        if image_positions:
            return max(image_positions)
        return int(valid_positions[0])

    def _predicted_category_matches(
        self,
        predicted_ids: torch.Tensor,
        gold_id: int,
        category_patterns: Sequence[Sequence[int]],
        action_positions: torch.Tensor,
    ) -> bool:
        if action_positions.numel() == 0:
            return False
        min_pos = int(action_positions[0])
        max_pos = int(action_positions[-1])
        tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            valid_ids = [
                int(token_id)
                for token_id in predicted_ids[min_pos : max_pos + 1].detach().cpu().tolist()
                if int(token_id) >= 0
            ]
            if valid_ids:
                predicted_text = tokenizer.decode(valid_ids, skip_special_tokens=False)
                predicted_categories = set(infer_action_categories_from_text(predicted_text))
                if 0 <= gold_id < len(self.action_priority) and self.action_priority[gold_id] in predicted_categories:
                    return True

        if not category_patterns:
            return False
        valid_positions = torch.arange(min_pos, max_pos + 1, device=predicted_ids.device)
        for category_pattern in category_patterns:
            found = self._find_subsequence(predicted_ids, list(category_pattern), valid_positions)
            if found is not None:
                return True
        return False

    def _alignment_loss_for_category(
        self,
        row_logits: torch.Tensor,
        row_labels: torch.Tensor,
        category_patterns: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        action_positions = torch.nonzero(
            self._action_region_mask(row_labels.unsqueeze(0))[0],
            as_tuple=False,
        ).flatten()
        valid_positions = action_positions
        if valid_positions.numel() == 0:
            valid_positions = torch.nonzero(row_labels.ne(-100), as_tuple=False).flatten()

        start = None
        target_ids = None
        for category_pattern in category_patterns:
            start = self._find_subsequence(row_labels, list(category_pattern), valid_positions)
            if start is not None:
                target_ids = row_labels[start : start + len(category_pattern)]
                break
        if start is None:
            if action_positions.numel() == 0:
                return row_logits.new_zeros(())
            start = int(action_positions[0])
            target_ids = row_labels[start : start + 1]
            pred_logits = row_logits[start - 1 : start] if start > 0 else row_logits[start : start + 1]
        else:
            pred_start = max(start - 1, 0)
            pred_logits = row_logits[pred_start : pred_start + len(target_ids)]

        if pred_logits.shape[0] != target_ids.shape[0] or target_ids.numel() == 0:
            return row_logits.new_zeros(())
        return F.cross_entropy(
            pred_logits.float(),
            target_ids.to(device=row_logits.device),
            ignore_index=-100,
            reduction="mean",
        )

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
