"""Action normalization helpers for Minecraft VLA manifests.

The public Minecraft datasets use different action spaces. TESS uses Lumine
action chunks, CraftJarvis uses reserved-token action strings, and local smoke
examples may use structured keyboard/mouse commands. This module gives all of
them one outer schema without pretending that they are directly interchangeable.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


ACTION_SCHEMA_VERSION = "minecraft_vla_action_v1"
LUMINE_ACTION_RE = re.compile(r"^\s*<\|action_start\|>(?P<body>.*)<\|action_end\|>\s*$", re.DOTALL)
CRAFTJARVIS_TOKEN_RE = re.compile(r"<\|reserved_special_token_\d+\|>")
MINECRAFT_ACTION_PRIORITY = [
    "attack",
    "use",
    "place",
    "break",
    "craft",
    "equip",
    "jump",
    "sneak",
    "sprint",
    "camera",
    "move",
    "inventory",
    "noop",
    "other",
]
ACTION_CATEGORY_ALIASES = {
    "attack": ['"attack":1', '"attack": 1', "LMB", "left click", "mouse left"],
    "use": ['"use":1', '"use": 1', "RMB", "right click", "interact", "mouse right"],
    "place": ['"place":1', '"place": 1', "place_block"],
    "break": ['"break":1', '"break": 1', '"dig":1', '"dig": 1', "mine"],
    "craft": ['"craft":1', '"craft": 1', "nearbyCraft", "nearby_craft", "nearbySmelt", "nearby_smelt"],
    "equip": ['"equip":1', '"equip": 1'],
    "jump": ["jump", "space", "SPACE"],
    "sneak": ['"sneak":1', '"sneak": 1', "shift", "SHIFT"],
    "sprint": ['"sprint":1', '"sprint": 1', "ctrl", "CTRL"],
    "camera": ["camera", "pitch", "yaw"],
    "move": [
        '"forward":1',
        '"forward": 1',
        '"back":1',
        '"back": 1',
        '"left":1',
        '"left": 1',
        '"right":1',
        '"right": 1',
        "W",
        "A",
        "S",
        "D",
    ],
    "inventory": ['"inventory":1', '"inventory": 1', "Esc", "ESC"],
    "noop": ["noop", "no_op", "no-op"],
    "other": ["other"],
}


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


def _truthy_action_value(value: Any) -> bool:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return any(_truthy_action_value(item) for item in value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered not in {"", "0", "none", "false", "noop", "no_op"}
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return abs(float(value)) > 1e-6
    return value is not None


def _add_category(categories: List[str], category: str) -> None:
    if category and category not in categories:
        categories.append(category)


def _categories_from_command(command: Mapping[str, Any]) -> List[str]:
    categories: List[str] = []
    for raw_key, value in command.items():
        key = str(raw_key).strip().lower()
        if not _truthy_action_value(value):
            continue

        if key in {"attack"}:
            _add_category(categories, "attack")
        elif key in {"use", "interact"}:
            _add_category(categories, "use")
        elif key in {"place", "place_block"}:
            _add_category(categories, "place")
        elif key in {"break", "dig", "mine"}:
            _add_category(categories, "break")
        elif key in {"craft", "nearbycraft", "nearby_craft", "nearby_smelt"}:
            _add_category(categories, "craft")
        elif key in {"equip"}:
            _add_category(categories, "equip")
        elif key in {"jump"}:
            _add_category(categories, "jump")
        elif key in {"sneak"}:
            _add_category(categories, "sneak")
        elif key in {"sprint"}:
            _add_category(categories, "sprint")
        elif key in {"camera", "pitch", "yaw"}:
            _add_category(categories, "camera")
        elif key in {"forward", "back", "left", "right", "move"}:
            _add_category(categories, "move")
        elif key in {"inventory"}:
            _add_category(categories, "inventory")
        elif key in {"key", "keyboard"}:
            categories.extend(_categories_from_text(str(value)))
        elif key in {"button", "mouse"}:
            button = str(value).lower()
            if "left" in button:
                _add_category(categories, "attack")
            elif "right" in button:
                _add_category(categories, "use")

    return categories


def _categories_from_text(text: str) -> List[str]:
    lowered = str(text or "").lower()
    tokens = [part.strip().lower() for part in re.split(r"[;\s,|]+", str(text or "")) if part.strip()]
    categories: List[str] = []

    if any(term in lowered for term in ["attack", "mouse_press('left", 'button":"left', "left click", "left_click"]):
        _add_category(categories, "attack")
    if "lmb" in tokens:
        _add_category(categories, "attack")
    if any(term in lowered for term in ["use", "interact", "mouse_press('right", 'button":"right', "right click", "right_click"]):
        _add_category(categories, "use")
    if "rmb" in tokens:
        _add_category(categories, "use")
    if any(term in lowered for term in ["place", "place_block"]):
        _add_category(categories, "place")
    if any(term in lowered for term in ["break", "dig", "mine"]):
        _add_category(categories, "break")
    if any(term in lowered for term in ["craft", "nearbycraft", "nearby_craft", "nearby_smelt"]):
        _add_category(categories, "craft")
    if "equip" in lowered:
        _add_category(categories, "equip")
    if any(term in lowered for term in ["jump", 'key":"space', "key': 'space", "key: space", "press space"]):
        _add_category(categories, "jump")
    if "space" in tokens:
        _add_category(categories, "jump")
    if "sneak" in lowered:
        _add_category(categories, "sneak")
    if "shift" in tokens:
        _add_category(categories, "sneak")
    if "sprint" in lowered:
        _add_category(categories, "sprint")
    if "ctrl" in tokens or "control" in tokens:
        _add_category(categories, "sprint")
    if any(term in lowered for term in ["camera", "pitch", "yaw"]):
        _add_category(categories, "camera")
    if any(term in lowered for term in ["forward", "backward", "move", 'key":"w', 'key":"a', 'key":"s', 'key":"d']):
        _add_category(categories, "move")
    if any(token in {"w", "a", "s", "d"} for token in tokens):
        _add_category(categories, "move")
    if "inventory" in lowered:
        _add_category(categories, "inventory")
    if any(token in {"e", "esc", "escape"} for token in tokens):
        _add_category(categories, "inventory")
    if any(term in lowered for term in ["noop", "no_op", "no-op"]):
        _add_category(categories, "noop")

    return categories


def infer_action_categories_from_actions(actions: Iterable[Mapping[str, Any]]) -> List[str]:
    """Infer Minecraft action categories for paper-style priority matching.

    CombatVLA defines a priority sequence over action categories. The original
    Black Myth: Wukong categories are private, so Minecraft reproduction keeps
    the same matching formula but derives categories from the normalized
    Minecraft action schema.
    """

    categories: List[str] = []
    for action in actions:
        if not isinstance(action, Mapping):
            for category in _categories_from_text(str(action)):
                _add_category(categories, category)
            continue

        command = action.get("command")
        if isinstance(command, Mapping):
            for category in _categories_from_command(command):
                _add_category(categories, category)

        raw = action.get("raw")
        if raw is not None:
            for category in _categories_from_text(str(raw)):
                _add_category(categories, category)

        for token in action.get("tokens") or []:
            for category in _categories_from_text(str(token)):
                _add_category(categories, category)

    if not categories:
        categories.append("other")
    return categories


def _decode_first_json_object(text: str) -> Optional[Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"{", str(text or "")):
        try:
            value, _ = decoder.raw_decode(str(text)[match.start() :])
        except json.JSONDecodeError:
            continue
        return value
    return None


def infer_action_categories_from_text(text: str, *, parse_json: bool = True) -> List[str]:
    if parse_json:
        parsed = _decode_first_json_object(text)
        if isinstance(parsed, Mapping):
            actions = parsed.get("actions")
            if isinstance(actions, list):
                categories = infer_action_categories_from_actions(actions)
                if categories:
                    return categories
            categories = infer_action_categories_from_actions([parsed])
            if categories and categories != ["other"]:
                return categories

    categories = _categories_from_text(text)
    return categories or ["other"]


def highest_priority_action_category(
    categories: Sequence[str],
    priority: Sequence[str] = MINECRAFT_ACTION_PRIORITY,
) -> str:
    category_set = {str(category) for category in categories}
    for category in priority:
        if category in category_set:
            return category
    return str(categories[0]) if categories else "other"


def priority_alpha(
    category: str,
    priority: Sequence[str] = MINECRAFT_ACTION_PRIORITY,
    *,
    alpha_min: float = 0.1,
    alpha_max: float = 1.0,
) -> float:
    """Paper Eq. 5 alpha: exponential priority weights normalized to [0.1, 1.0]."""

    if not priority:
        return float(alpha_min)

    try:
        index = list(priority).index(category)
    except ValueError:
        index = len(priority) - 1

    count = len(priority)
    if count <= 1:
        return float(alpha_max)

    raw = 2 ** (count - index - 1)
    raw_min = 1
    raw_max = 2 ** (count - 1)
    normalized = (raw - raw_min) / (raw_max - raw_min)
    return float(alpha_min + normalized * (alpha_max - alpha_min))
