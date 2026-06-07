"""Action normalization helpers for Minecraft VLA manifests.

The public Minecraft datasets use different action spaces. TESS uses Lumine
action chunks, CraftJarvis uses reserved-token action strings, and local smoke
examples may use structured keyboard/mouse commands. This module gives all of
them one outer schema without pretending that they are directly interchangeable.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional


ACTION_SCHEMA_VERSION = "minecraft_vla_action_v1"
LUMINE_ACTION_RE = re.compile(r"^\s*<\|action_start\|>(?P<body>.*)<\|action_end\|>\s*$", re.DOTALL)
CRAFTJARVIS_TOKEN_RE = re.compile(r"<\|reserved_special_token_\d+\|>")


def _clean_raw_text(value: Any) -> str:
    return str(value or "").strip()


def infer_source_format(raw: str) -> str:
    """Infer a conservative source action format from a raw action string."""

    if LUMINE_ACTION_RE.match(raw):
        return "lumine_action_chunk"
    if CRAFTJARVIS_TOKEN_RE.search(raw):
        return "craftjarvis_reserved_tokens"
    return "raw_text"


def parse_lumine_tokens(raw: str) -> List[str]:
    """Split a Lumine action chunk into its semicolon-separated fields."""

    match = LUMINE_ACTION_RE.match(raw)
    if not match:
        return []
    return [part.strip() for part in match.group("body").split(";")]


def normalize_action(
    action: Any,
    *,
    source_format: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize a dataset-specific action into the shared manifest schema.

    The normalized object intentionally keeps the source format and raw payload.
    That lets Stage1/2/3 share a target envelope while avoiding unsupported
    translations between Lumine chunks, CraftJarvis reserved tokens, and local
    keyboard/mouse commands.
    """

    if isinstance(action, Mapping):
        if action.get("schema") == ACTION_SCHEMA_VERSION:
            normalized = dict(action)
        elif "raw" in action:
            raw = _clean_raw_text(action.get("raw"))
            normalized = {
                "schema": ACTION_SCHEMA_VERSION,
                "source_format": source_format or str(action.get("source_format") or infer_source_format(raw)),
                "raw": raw,
            }
        else:
            normalized = {
                "schema": ACTION_SCHEMA_VERSION,
                "source_format": source_format or "framework_control_json",
                "command": dict(action),
            }
    else:
        raw = _clean_raw_text(action)
        normalized = {
            "schema": ACTION_SCHEMA_VERSION,
            "source_format": source_format or infer_source_format(raw),
            "raw": raw,
        }

    if source and "source" not in normalized:
        normalized["source"] = source

    raw_value = normalized.get("raw")
    if normalized.get("source_format") == "lumine_action_chunk" and isinstance(raw_value, str):
        tokens = parse_lumine_tokens(raw_value)
        if tokens:
            normalized["tokens"] = tokens

    return normalized


def normalize_actions(
    actions: Iterable[Any],
    *,
    source_format: Optional[str] = None,
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return [normalize_action(action, source_format=source_format, source=source) for action in actions]


def normalize_minerl_action(action: Mapping[str, Any], *, source: str) -> Dict[str, Any]:
    """Normalize a MineRL/Optimus frame-level action dictionary."""

    def scalar(value: Any) -> Any:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list) and len(value) == 1:
            return scalar(value[0])
        return value

    command = {str(key): scalar(value) for key, value in action.items()}
    return {
        "schema": ACTION_SCHEMA_VERSION,
        "source_format": "minerl_frame_action",
        "source": source,
        "command": command,
    }
