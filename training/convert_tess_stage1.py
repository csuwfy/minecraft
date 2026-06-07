"""Convert TESS Minecraft VLA Stage 1 into CombatVLA stage-1 manifests."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Tuple

import pyarrow.parquet as pq
from datasets import load_dataset

from training.actions import normalize_action
from training.image_refs import TESS_STAGE1_SCHEME


def decode_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, Mapping):
        raw = value.get("bytes")
        if isinstance(raw, bytes):
            return raw
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def write_record(handle, frames: Iterable[str], sample: Mapping[str, Any], task: str) -> None:
    record = {
        "frames": list(frames),
        "task": task,
        "actions": [
            normalize_action(
                sample.get("action"),
                source_format="lumine_action_chunk",
                source="TESS-Computer/minecraft-vla-stage1",
            )
        ],
        "reasoning": "",
        "metadata": {
            "source": "TESS-Computer/minecraft-vla-stage1",
            "stage_objective": "coarse_video_action_pretraining",
            "video_id": sample.get("video_id"),
            "frame_idx": sample.get("frame_idx"),
        },
    }
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def selected_parquet_files(args) -> List[Path]:
    data_dir = Path(args.data_dir)
    data_files = sorted((data_dir / "data").glob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir / 'data'}")
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, shard-count)")
    selected = [path for index, path in enumerate(data_files) if index % args.shard_count == args.shard_index]
    if not selected:
        raise FileNotFoundError(
            f"No parquet files selected for shard {args.shard_index}/{args.shard_count} under {data_dir / 'data'}"
        )
    return selected


def load_source(args):
    if args.data_dir:
        data_files = [str(path) for path in selected_parquet_files(args)]
        return load_dataset("parquet", data_files=data_files, split="train", streaming=args.streaming)
    return load_dataset(args.dataset, split=args.split, streaming=args.streaming)


def iter_parquet_refs(args) -> Tuple[List[Path], Iterable[Tuple[Mapping[str, Any], str]]]:
    if not args.data_dir:
        raise ValueError("--reference-parquet-frames requires --data-dir")
    data_files = selected_parquet_files(args)

    def iterator():
        for file_index, path in enumerate(data_files):
            table = pq.read_table(path, columns=["video_id", "frame_idx", "action"])
            names = table.schema.names
            video_col = table.column("video_id") if "video_id" in names else None
            frame_col = table.column("frame_idx") if "frame_idx" in names else None
            action_col = table.column("action") if "action" in names else None
            if video_col is None or frame_col is None or action_col is None:
                raise ValueError(f"Missing required columns in {path}")
            for row_index in range(table.num_rows):
                sample = {
                    "video_id": video_col[row_index].as_py(),
                    "frame_idx": frame_col[row_index].as_py(),
                    "action": action_col[row_index].as_py(),
                }
                yield sample, f"{TESS_STAGE1_SCHEME}{file_index}/{row_index}"

    return data_files, iterator()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TESS stage1 to CombatVLA stage1 JSONL.")
    parser.add_argument("--dataset", default="TESS-Computer/minecraft-vla-stage1")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-count", type=int, default=100000)
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument("--val-every", type=int, default=20, help="Write every Nth eligible sample to validation until val-count is reached.")
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--task", default="Play Minecraft.")
    parser.add_argument(
        "--frame-prefix",
        default=None,
        help="Optional subdirectory under frames_tess_stage1. Defaults to shard_XXXX when sharded.",
    )
    parser.add_argument(
        "--overwrite-frames",
        action="store_true",
        help="Write frame files directly instead of checking whether they already exist.",
    )
    parser.add_argument(
        "--reference-parquet-frames",
        action="store_true",
        help="Write TESS parquet frame references instead of copying image bytes to frame files.",
    )
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    frames_root = output_root / "frames_tess_stage1"
    if not args.reference_parquet_frames:
        frames_root.mkdir(parents=True, exist_ok=True)
    frame_prefix = args.frame_prefix
    if frame_prefix is None and args.shard_count > 1:
        frame_prefix = f"shard_{args.shard_index:04d}"
    frame_prefix = (frame_prefix or "").strip("/")

    train_path = output_root / "train_stage1.jsonl"
    val_path = output_root / "val_stage1.jsonl"

    windows: Dict[str, Deque[str]] = defaultdict(lambda: deque(maxlen=args.max_frames))
    last_frame_idx: Dict[str, int] = {}
    written_train = 0
    written_val = 0
    eligible = 0

    tess_stage1_index: Optional[str] = None
    if args.reference_parquet_frames:
        parquet_files, dataset_iter = iter_parquet_refs(args)
        index_path = output_root / "tess_stage1_parquet_index.json"
        index = {
            "format": "tess_stage1_parquet_refs",
            "source": "TESS-Computer/minecraft-vla-stage1",
            "parquet_files": [str(path.resolve()) for path in parquet_files],
        }
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tess_stage1_index = str(index_path)
    else:
        dataset_iter = ((sample, None) for sample in load_source(args))

    with train_path.open("w", encoding="utf-8") as train_handle, val_path.open("w", encoding="utf-8") as val_handle:
        for sample, frame_ref in dataset_iter:
            video_id = str(sample.get("video_id") or "unknown_video")
            frame_idx = int(sample.get("frame_idx") or 0)
            action = str(sample.get("action") or "").strip()
            if not action:
                continue

            previous = last_frame_idx.get(video_id)
            if previous is not None and frame_idx <= previous:
                windows[video_id].clear()
            last_frame_idx[video_id] = frame_idx

            if frame_ref is not None:
                frame_rel = frame_ref
            else:
                if frame_prefix:
                    frame_rel = f"frames_tess_stage1/{frame_prefix}/{video_id}/frame_{frame_idx:08d}.jpg"
                else:
                    frame_rel = f"frames_tess_stage1/{video_id}/frame_{frame_idx:08d}.jpg"
                frame_path = output_root / frame_rel
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                if args.overwrite_frames or not frame_path.exists():
                    frame_path.write_bytes(decode_bytes(sample.get("image")))

            windows[video_id].append(frame_rel)
            if len(windows[video_id]) < args.max_frames:
                continue

            eligible += 1
            write_val = args.val_every > 0 and eligible % args.val_every == 0 and written_val < args.val_count
            if write_val:
                write_record(val_handle, windows[video_id], sample, args.task)
                written_val += 1
            elif written_train < args.train_count:
                write_record(train_handle, windows[video_id], sample, args.task)
                written_train += 1
            elif written_val < args.val_count:
                write_record(val_handle, windows[video_id], sample, args.task)
                written_val += 1
            else:
                break

    print(f"train={written_train} val={written_val}")
    if tess_stage1_index:
        print(tess_stage1_index)
    print(train_path)
    print(val_path)


if __name__ == "__main__":
    main()
