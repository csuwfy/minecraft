"""Dataset loading for Minecraft VLA supervised fine-tuning."""

from __future__ import annotations

import json
import os
from array import array
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union

from torch.utils.data import Dataset

from training.aot import AOTExample, build_messages
from training.image_refs import is_tess_stage1_ref


def manifest_record_count(manifest_path: Union[str, Path]) -> int:
    path = Path(manifest_path)
    cache_path = path.with_name(path.name + ".offsets.u64")
    meta_path = path.with_name(path.name + ".offsets.json")
    stat = path.stat()
    if cache_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("path") == str(path.resolve())
                and meta.get("size") == stat.st_size
                and meta.get("mtime_ns") == stat.st_mtime_ns
                and meta.get("format") == "uint64_offsets_v1"
                and isinstance(meta.get("records"), int)
            ):
                return int(meta["records"])
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


class MinecraftVLADataset(Dataset):
    """JSONL manifest dataset.

    Each line is expected to contain:

    {
      "frames": ["relative/or/absolute/frame_0001.png", "..."],
      "task": "collect wood",
      "actions": [{"type": "keyboard", "action": "press", "key": "w"}],
      "reasoning": "optional AoT text"
    }
    """

    def __init__(
        self,
        manifest_path: str,
        image_root: Optional[str] = None,
        stage: int = 3,
        max_frames: int = 3,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        tess_stage1_index: Optional[str] = None,
        require_reasoning: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root) if image_root else self.manifest_path.parent
        self.stage = stage
        self.max_frames = max_frames
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.tess_stage1_index = tess_stage1_index
        self.require_reasoning = require_reasoning
        self._offsets = self._load_or_build_offsets()

        if not self._offsets:
            raise ValueError(f"No records found in {self.manifest_path}")

    def _load_or_build_offsets(self) -> List[int]:
        cache_enabled = os.environ.get("VLA_OFFSET_CACHE", "1") != "0"
        if not cache_enabled:
            return self._build_offsets()

        cache_path = self.manifest_path.with_name(self.manifest_path.name + ".offsets.u64")
        meta_path = self.manifest_path.with_name(self.manifest_path.name + ".offsets.json")
        stat = self.manifest_path.stat()
        expected = {
            "path": str(self.manifest_path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "format": "uint64_offsets_v1",
        }

        if cache_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if all(meta.get(key) == value for key, value in expected.items()):
                    offsets = array("Q")
                    with cache_path.open("rb") as handle:
                        offsets.fromfile(handle, cache_path.stat().st_size // offsets.itemsize)
                    print(f"Loaded {len(offsets):,} manifest offsets from {cache_path}", flush=True)
                    return list(offsets)
            except (OSError, ValueError, json.JSONDecodeError):
                pass

        offsets = self._build_offsets()
        tmp_cache = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
        offset_array = array("Q", offsets)
        with tmp_cache.open("wb") as handle:
            offset_array.tofile(handle)
        expected["records"] = len(offsets)
        tmp_meta.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_cache, cache_path)
        os.replace(tmp_meta, meta_path)
        print(f"Wrote {len(offsets):,} manifest offsets to {cache_path}", flush=True)
        return offsets

    def _build_offsets(self) -> List[int]:
        offsets: List[int] = []
        next_report = 1_000_000
        with self.manifest_path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
                    if len(offsets) >= next_report:
                        print(
                            f"Indexed {len(offsets):,} records from {self.manifest_path}",
                            flush=True,
                        )
                        next_report += 1_000_000
        return offsets

    def _iter_records(self) -> Iterator[Dict[str, Any]]:
        with self.manifest_path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {self.manifest_path}:{line_number}") from exc
                yield self._normalize_record(record, line_number)

    def _normalize_record(self, record: Mapping[str, Any], line_number: int) -> Dict[str, Any]:
        example = AOTExample.from_record(record, stage=self.stage)
        if not example.frames:
            raise ValueError(f"Record {line_number} has no frames.")
        if not example.actions:
            raise ValueError(f"Record {line_number} has no actions.")
        if self.require_reasoning and not example.reasoning.strip():
            metadata = example.metadata or {}
            source = metadata.get("source") if isinstance(metadata, Mapping) else None
            raise ValueError(
                f"Record {line_number} in {self.manifest_path} has empty reasoning. "
                "Paper-style AoT training requires non-empty reasoning; generate and merge "
                f"teacher/human reasoning first. source={source!r}"
            )

        frames = [self._resolve_image_path(path) for path in example.frames[: self.max_frames]]
        normalized = {
            "frames": frames,
            "task": example.task,
            "actions": example.actions,
            "reasoning": example.reasoning,
            "metadata": example.metadata,
            "tess_stage1_index": record.get("tess_stage1_index") or self.tess_stage1_index,
        }
        return normalized

    def _resolve_image_path(self, path: str) -> str:
        if is_tess_stage1_ref(path):
            return path
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.image_root / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Frame not found: {candidate}")
        return os.fspath(candidate)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self._read_record(index)
        example = AOTExample.from_record(record, stage=self.stage)
        kwargs = {}
        if self.system_prompt is not None:
            kwargs["system_prompt"] = self.system_prompt
        if self.user_prompt is not None:
            kwargs["user_prompt"] = self.user_prompt
        return {
            "messages": build_messages(example, **kwargs),
            "frames": example.frames,
            "task": example.task,
            "actions": list(example.actions),
            "tess_stage1_index": record.get("tess_stage1_index") or self.tess_stage1_index,
        }

    def _read_record(self, index: int) -> Dict[str, Any]:
        offset = self._offsets[index]
        with self.manifest_path.open("rb") as handle:
            handle.seek(offset)
            raw = handle.readline()
        try:
            record = json.loads(raw.decode("utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {self.manifest_path} offset {offset}") from exc
        return self._normalize_record(record, index + 1)
