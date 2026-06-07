"""Action-of-Thought formatting helpers.

The CombatVLA paper uses stage-specific Action-of-Thought targets: Stage 1/2
place reasoning before the action JSON, while Stage 3 places action JSON before
the truncated reasoning marker. This module keeps the format explicit so
datasets can be generated and audited before training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

from training.actions import ACTION_SCHEMA_VERSION, normalize_actions

TRUNC_TOKEN = "<TRUNC>"


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Minecraft agent. Predict the next low-level actions "
    "from the visual observation sequence and task context."
)


DEFAULT_USER_PROMPT = (
    "Given the frame sequence from Minecraft, predict the next action. "
    f"Return a compact JSON object using schema {ACTION_SCHEMA_VERSION}."
)


STAGE_USER_PROMPTS = {
    1: (
        "Given the coarse Minecraft frame sequence, predict the next action chunk. "
        f"Return concise reasoning followed by a compact JSON object using schema {ACTION_SCHEMA_VERSION}."
    ),
    2: (
        "Given the recent Minecraft frames and task, predict the next aligned action. "
        f"Return concise reasoning followed by a compact JSON object using schema {ACTION_SCHEMA_VERSION}."
    ),
    3: (
        "Given the recent Minecraft frames and task, predict the next aligned action. "
        f"First return a compact JSON object using schema {ACTION_SCHEMA_VERSION}, then write {TRUNC_TOKEN} "
        "and concise reasoning."
    ),
}


@dataclass(frozen=True)
class AOTExample:
    """A normalized training example before chat-template conversion."""

    frames: List[str]
    task: str
    actions: List[Mapping[str, Any]]
    reasoning: str = ""
    stage: int = 3
    metadata: Optional[Mapping[str, Any]] = None

    @classmethod
    def from_record(cls, record: Mapping[str, Any], stage: int) -> "AOTExample":
        frames = record.get("frames") or record.get("images") or []
        if isinstance(frames, str):
            frames = [frames]

        source = None
        metadata = record.get("metadata")
        if isinstance(metadata, Mapping):
            source = metadata.get("source")

        actions = record.get("actions")
        if actions is None:
            action = record.get("action")
            actions = [action] if action is not None else []

        if not isinstance(actions, list):
            raise ValueError("Record field 'actions' must be a list.")

        return cls(
            frames=[str(path) for path in frames],
            task=str(record.get("task") or record.get("instruction") or ""),
            actions=normalize_actions(actions, source=source),
            reasoning=str(record.get("reasoning") or record.get("analysis") or ""),
            stage=stage,
            metadata=metadata,
        )


def action_json(actions: Iterable[Mapping[str, Any]]) -> str:
    """Serialize normalized action commands for SFT targets."""

    return json.dumps({"actions": list(actions)}, ensure_ascii=False, separators=(",", ":"))


def target_text(example: AOTExample) -> str:
    """Build the assistant target for a CombatVLA-style AoT stage."""

    actions = action_json(example.actions)

    if example.stage == 1:
        if example.reasoning:
            return f"{example.reasoning}\n{actions}"
        return actions

    if example.stage == 2:
        if example.reasoning:
            return f"{example.reasoning}\n{actions}"
        return actions

    if example.stage == 3:
        if example.reasoning:
            return f"{actions}{TRUNC_TOKEN}{example.reasoning}"
        return f"{actions}{TRUNC_TOKEN}"

    raise ValueError(f"Unsupported AoT stage: {example.stage}")


def build_user_text(example: AOTExample, base_prompt: str = DEFAULT_USER_PROMPT) -> str:
    if base_prompt == DEFAULT_USER_PROMPT:
        base_prompt = STAGE_USER_PROMPTS.get(example.stage, DEFAULT_USER_PROMPT)
    task = example.task.strip()
    if task:
        return f"{base_prompt}\nTask: {task}"
    return base_prompt


def build_messages(
    example: AOTExample,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt: str = DEFAULT_USER_PROMPT,
) -> List[Dict[str, Any]]:
    """Create OpenAI-style multimodal chat messages.

    Hugging Face processors for Qwen2.5-VL and similar models can consume this
    structure when applying a chat template.
    """

    content: List[Dict[str, Any]] = [{"type": "text", "text": build_user_text(example, user_prompt)}]
    content.extend({"type": "image", "image": frame} for frame in example.frames)

    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": target_text(example)}]},
    ]
