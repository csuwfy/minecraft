"""Convert CraftJarvis Minecraft VLA SFT data into CombatVLA supplement manifests.

CraftJarvis rows are single image/action samples. The action is represented as
reserved special tokens, not the Lumine action chunks used by TESS, so this
converter writes separate supplement manifests instead of mixing them with TESS
files by default.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Mapping, Optional, Tuple

from datasets import load_dataset

from training.actions import normalize_action


OBSERVATION_RE = re.compile(r"\n\s*observation\s*:\s*$", re.IGNORECASE)
SAFE_PATH_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def split_sample_id(sample_id: str) -> Tuple[str, int]:
    prefix, _, suffix = sample_id.rpartition("_")
    if suffix.isdigit():
        return prefix, int(suffix)
    return sample_id, -1


def safe_path_part(value: str) -> str:
    return SAFE_PATH_RE.sub("_", value)


def decode_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, list) and value:
        return decode_bytes(value[0])
    if isinstance(value, Mapping):
        raw = value.get("bytes")
        if isinstance(raw, bytes):
            return raw
    raise TypeError(f"Unsupported image_bytes value: {type(value)!r}")


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return ""


def extract_instruction_and_action(sample: Mapping[str, Any]) -> Tuple[str, str]:
    instruction = ""
    action = ""
    for message in sample.get("conversations") or []:
        role = message.get("role")
        text = text_from_content(message.get("content")).strip()
        if role == "user" and text:
            instruction = OBSERVATION_RE.sub("", text).strip()
        elif role == "assistant" and text:
            action = text.strip()
    return instruction, action


def label_value(labels: Iterable[str], prefix: str) -> Optional[str]:
    for label in labels:
        if str(label).startswith(prefix):
            return str(label)[len(prefix) :]
    return None


def metadata_from_sample(sample: Mapping[str, Any], segment_id: str, frame_idx: int) -> Mapping[str, Any]:
    labels = [str(x) for x in (sample.get("label") or [])]
    return {
        "source": "CraftJarvis/minecraft-vla-sft",
        "id": sample.get("id"),
        "segment_id": segment_id,
        "frame_idx": frame_idx,
        "labels": labels,
        "task_category": labels[2] if len(labels) > 2 else None,
        "horizon": label_value(labels, "h="),
        "continuation": label_value(labels, "c="),
    }


def write_record(
    handle,
    frames: Iterable[str],
    task: str,
    action: str,
    metadata: Mapping[str, Any],
) -> None:
    record = {
        "frames": list(frames),
        "task": task,
        "actions": [
            normalize_action(
                action,
                source_format="craftjarvis_reserved_tokens",
                source="CraftJarvis/minecraft-vla-sft",
            )
        ],
        "reasoning": "",
        "metadata": metadata,
    }
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def iter_source_dataset(args):
    if args.data_dir:
        data_dir = Path(args.data_dir)
        data_files = sorted(str(path) for path in (data_dir / "data").glob(f"{args.split}-*.parquet"))
        if not data_files:
            data_files = sorted(str(path) for path in (data_dir / "data").glob("*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"No parquet files found under {data_dir / 'data'}")
        return load_dataset("parquet", data_files=data_files, split="train", streaming=args.streaming)
    return load_dataset(args.dataset, split=args.split, streaming=args.streaming)


def build_index(args, output_root: Path) -> Path:
    index_path = output_root / "craftjarvis_index.sqlite3"
    if index_path.exists() and not args.rebuild_index:
        print(f"Using existing index: {index_path}")
        return index_path

    if index_path.exists():
        index_path.unlink()

    conn = sqlite3.connect(index_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE records (
            source_index INTEGER PRIMARY KEY,
            segment_id TEXT NOT NULL,
            frame_idx INTEGER NOT NULL,
            frame_rel TEXT NOT NULL,
            task TEXT NOT NULL,
            action TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        )
        """
    )

    dataset = iter_source_dataset(args)
    inserted = 0
    for source_index, sample in enumerate(dataset):
        if args.max_source_rows and source_index >= args.max_source_rows:
            break

        sample_id = str(sample.get("id") or "")
        segment_id, frame_idx = split_sample_id(sample_id)
        task, action = extract_instruction_and_action(sample)
        if not task or not action:
            continue

        safe_segment = safe_path_part(segment_id)
        safe_frame_idx = frame_idx if frame_idx >= 0 else source_index
        frame_rel = f"frames_craftjarvis/{safe_segment}/frame_{safe_frame_idx:08d}.jpg"
        frame_path = output_root / frame_rel
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        if not frame_path.exists():
            frame_path.write_bytes(decode_bytes(sample.get("image_bytes")))

        metadata = metadata_from_sample(sample, segment_id, frame_idx)
        conn.execute(
            """
            INSERT INTO records (
                source_index, segment_id, frame_idx, frame_rel, task, action, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_index,
                segment_id,
                frame_idx,
                frame_rel,
                task,
                action,
                json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
            ),
        )
        inserted += 1
        if inserted % args.commit_interval == 0:
            conn.commit()
            print(f"indexed={inserted} source_index={source_index}", flush=True)

    conn.commit()
    conn.execute("CREATE INDEX idx_records_order ON records(segment_id, frame_idx, source_index)")
    conn.commit()
    conn.close()
    print(f"Indexed records: {inserted}")
    return index_path


def write_manifests(args, output_root: Path, index_path: Path) -> Mapping[str, int]:
    handles = {
        "train_stage2_craftjarvis": (output_root / "train_stage2_craftjarvis.jsonl").open(
            "w", encoding="utf-8"
        ),
        "val_stage2_craftjarvis": (output_root / "val_stage2_craftjarvis.jsonl").open("w", encoding="utf-8"),
    }
    if args.emit_stage1:
        handles["train_stage1_craftjarvis"] = (output_root / "train_stage1_craftjarvis.jsonl").open(
            "w", encoding="utf-8"
        )
        handles["val_stage1_craftjarvis"] = (output_root / "val_stage1_craftjarvis.jsonl").open(
            "w", encoding="utf-8"
        )
    counts = {key: 0 for key in handles}

    stage1_window: Deque[str] = deque(maxlen=args.stage1_frames)
    stage23_window: Deque[str] = deque(maxlen=args.stage23_frames)
    current_segment: Optional[str] = None

    conn = sqlite3.connect(index_path)
    try:
        cursor = conn.execute(
            """
            SELECT segment_id, frame_idx, frame_rel, task, action, metadata_json
            FROM records
            ORDER BY segment_id, frame_idx, source_index
            """
        )
        for segment_id, _frame_idx, frame_rel, task, action, metadata_json in cursor:
            if segment_id != current_segment:
                current_segment = segment_id
                stage1_window.clear()
                stage23_window.clear()

            stage1_window.append(frame_rel)
            stage23_window.append(frame_rel)
            metadata = json.loads(metadata_json)

            if len(stage1_window) >= args.stage1_frames:
                if args.emit_stage1 and counts["train_stage1_craftjarvis"] < args.train_count:
                    write_record(handles["train_stage1_craftjarvis"], stage1_window, task, action, metadata)
                    counts["train_stage1_craftjarvis"] += 1
                elif args.emit_stage1 and counts["val_stage1_craftjarvis"] < args.val_count:
                    write_record(handles["val_stage1_craftjarvis"], stage1_window, task, action, metadata)
                    counts["val_stage1_craftjarvis"] += 1

            if len(stage23_window) >= args.stage23_frames:
                if counts["train_stage2_craftjarvis"] < args.train_count:
                    write_record(handles["train_stage2_craftjarvis"], stage23_window, task, action, metadata)
                    counts["train_stage2_craftjarvis"] += 1
                elif counts["val_stage2_craftjarvis"] < args.val_count:
                    write_record(handles["val_stage2_craftjarvis"], stage23_window, task, action, metadata)
                    counts["val_stage2_craftjarvis"] += 1

            if (
                (not args.emit_stage1 or counts["train_stage1_craftjarvis"] >= args.train_count)
                and (not args.emit_stage1 or counts["val_stage1_craftjarvis"] >= args.val_count)
                and counts["train_stage2_craftjarvis"] >= args.train_count
                and counts["val_stage2_craftjarvis"] >= args.val_count
            ):
                break
    finally:
        conn.close()
        for handle in handles.values():
            handle.close()

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CraftJarvis VLA SFT data to CombatVLA manifests.")
    parser.add_argument("--dataset", default="CraftJarvis/minecraft-vla-sft")
    parser.add_argument("--data-dir", default=None, help="Local Hugging Face dataset directory from huggingface-cli.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-count", type=int, default=100000)
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument("--stage1-frames", type=int, default=20)
    parser.add_argument("--stage23-frames", type=int, default=4)
    parser.add_argument("--emit-stage1", action="store_true", help="Also write separate CraftJarvis Stage1 supplement manifests.")
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--max-source-rows", type=int, default=0)
    parser.add_argument("--commit-interval", type=int, default=10000)
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    frames_root = output_root / "frames_craftjarvis"
    output_root.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)

    index_path = build_index(args, output_root)
    counts = write_manifests(args, output_root, index_path)
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
