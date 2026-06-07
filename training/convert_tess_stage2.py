"""Convert TESS Minecraft VLA Stage 2 samples into JSONL manifests.

The source dataset stores one JPEG frame per row with an action string and an
instruction that is only present at segment starts. This converter keeps a
rolling window of recent frames per video and emits CombatVLA-style records.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Mapping

from datasets import load_dataset

from training.actions import normalize_action


def decode_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, Mapping):
        raw = value.get("bytes")
        if isinstance(raw, bytes):
            return raw
    raise TypeError(f"Unsupported image_bytes value: {type(value)!r}")


def write_record(handle, frames: Iterable[str], sample: Mapping[str, Any], task: str) -> None:
    action = normalize_action(
        sample.get("action"),
        source_format="lumine_action_chunk",
        source="TESS-Computer/minecraft-vla-stage2",
    )
    record = {
        "frames": list(frames),
        "task": task,
        "actions": [action],
        "reasoning": "",
        "metadata": {
            "source": "TESS-Computer/minecraft-vla-stage2",
            "stage_objective": "fine_grained_instruction_action_alignment",
            "id": sample.get("id"),
            "video_id": sample.get("video_id"),
            "frame_idx": sample.get("frame_idx"),
            "task_category": sample.get("task_category"),
            "task_group": sample.get("task_group"),
            "target": sample.get("target"),
            "subset": sample.get("subset"),
            "is_segment_start": sample.get("is_segment_start"),
        },
    }
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TESS stage2 to CombatVLA JSONL manifests.")
    parser.add_argument("--dataset", default="TESS-Computer/minecraft-vla-stage2")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-count", type=int, default=10000)
    parser.add_argument("--val-count", type=int, default=256)
    parser.add_argument("--val-every", type=int, default=20, help="Write every Nth eligible sample to validation until val-count is reached.")
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    frames_root = output_root / "frames_tess_stage2"
    output_root.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)

    train_path = output_root / "train_stage2.jsonl"
    val_path = output_root / "val_stage2.jsonl"

    windows: Dict[str, Deque[str]] = defaultdict(lambda: deque(maxlen=args.max_frames))
    tasks: Dict[str, str] = defaultdict(str)
    written_train = 0
    written_val = 0
    eligible = 0

    if args.data_dir:
        data_dir = Path(args.data_dir)
        data_files = sorted(str(path) for path in (data_dir / "data").glob("*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"No parquet files found under {data_dir / 'data'}")
        dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=args.streaming)
    else:
        dataset = load_dataset(args.dataset, split=args.split, streaming=args.streaming)

    with train_path.open("w", encoding="utf-8") as train_handle, val_path.open("w", encoding="utf-8") as val_handle:
        for sample in dataset:
            video_id = str(sample.get("video_id") or "unknown_video")
            frame_idx = int(sample.get("frame_idx") or 0)
            instruction = str(sample.get("instruction") or "").strip()
            if instruction:
                tasks[video_id] = instruction
                windows[video_id].clear()

            frame_rel = f"frames_tess_stage2/{video_id}/frame_{frame_idx:08d}.jpg"
            frame_path = output_root / frame_rel
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            if not frame_path.exists():
                frame_path.write_bytes(decode_bytes(sample.get("image_bytes")))

            windows[video_id].append(frame_rel)
            if len(windows[video_id]) < args.max_frames:
                continue

            task = tasks[video_id] or instruction
            if not task:
                continue

            eligible += 1
            write_val = args.val_every > 0 and eligible % args.val_every == 0 and written_val < args.val_count
            if write_val:
                write_record(val_handle, windows[video_id], sample, task)
                written_val += 1
            elif written_train < args.train_count:
                write_record(train_handle, windows[video_id], sample, task)
                written_train += 1
            elif written_val < args.val_count:
                write_record(val_handle, windows[video_id], sample, task)
                written_val += 1
            else:
                break

    print(f"train={written_train} val={written_val}")
    print(train_path)
    print(val_path)


if __name__ == "__main__":
    main()
