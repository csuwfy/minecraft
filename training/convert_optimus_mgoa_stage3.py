"""Convert Optimus-2-MGOA trajectories into Minecraft Stage3 manifests.

This converter expects the full Optimus video split archive to be assembled as
`video.tar.gz`, plus `action.tar.gz` and `task_description_map.json`. Optimus
does not ship human-written Action-of-Thought explanations, so reasoning is
empty by default unless an explicit reasoning source is supplied.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import cv2

from training.actions import normalize_minerl_action


SOURCE = "iLearn-Lab/Optimus-2-MGOA"
ACTION_KEYS = ("attack", "use", "forward", "back", "left", "right", "jump", "sneak", "sprint")


def scalar(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1:
        return scalar(value[0])
    return value


def action_summary(action: Mapping[str, Any]) -> str:
    command = {str(key): scalar(value) for key, value in action.items()}
    active = [key for key in ACTION_KEYS if command.get(key) not in (0, 0.0, False, None)]
    camera = command.get("camera")
    if isinstance(camera, list) and camera and isinstance(camera[0], list):
        camera = camera[0]
    camera_text = ""
    if isinstance(camera, list) and len(camera) >= 2:
        dx = float(camera[0])
        dy = float(camera[1])
        if abs(dx) > 0.01 or abs(dy) > 0.01:
            camera_text = f" and adjusts the camera by [{dx:.2f}, {dy:.2f}]"
    if active:
        return "uses " + ", ".join(active) + camera_text
    if camera_text:
        return "keeps movement keys idle" + camera_text
    return "keeps a neutral action"


def task_text(task: str) -> str:
    return task.replace("_", " ").strip()


def reasoning_for(task: str, action: Mapping[str, Any]) -> str:
    return (
        f"The current goal is to {task_text(task)}. The selected action {action_summary(action)} "
        "to continue the trajectory toward that goal."
    )


def load_task_map(path: Path) -> Dict[str, str]:
    return {str(key): str(value) for key, value in json.loads(path.read_text(encoding="utf-8")).items()}


def extract_action_pickles(action_tar: Path, keys: set[str], actions_dir: Path) -> set[str]:
    actions_dir.mkdir(parents=True, exist_ok=True)
    available: set[str] = set()
    with tarfile.open(action_tar, "r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.startswith("action/"):
                continue
            path = Path(member.name)
            if path.suffix and path.suffix != ".pkl":
                continue
            key = path.stem
            if key not in keys:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            with (actions_dir / f"{key}.pkl").open("wb") as out:
                shutil.copyfileobj(handle, out, length=1024 * 1024)
            available.add(key)
    return available


def load_actions_from_file(actions_dir: Path, key: str) -> List[Mapping[str, Any]]:
    with (actions_dir / f"{key}.pkl").open("rb") as handle:
        return pickle.load(handle)


def write_frames(
    video_path: Path,
    frames_dir: Path,
    *,
    stride: int,
    max_frames: int,
) -> List[str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames_dir.mkdir(parents=True, exist_ok=True)
    rel_names: List[str] = []
    source_index = 0
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if source_index % stride == 0:
                out = frames_dir / f"frame_{source_index:08d}.jpg"
                if not out.exists():
                    cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                rel_names.append(out.name)
                written += 1
                if max_frames and written >= max_frames:
                    break
            source_index += 1
    finally:
        cap.release()

    return rel_names


def reasoning_key(trajectory_id: str, action_index: int) -> str:
    return f"{trajectory_id}:{action_index}"


def load_reasoning_jsonl(path: Path) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            trajectory_id = record.get("trajectory_id") or record.get("key") or record.get("video_id")
            action_index = first_present(record, "action_index", "frame_idx", "video_frame_index")
            reasoning = str(record.get("reasoning") or record.get("analysis") or "").strip()
            if trajectory_id is None or action_index is None:
                raise ValueError(f"{path}:{line_no} requires trajectory_id/key/video_id and action_index/frame_idx")
            if reasoning:
                lookup[reasoning_key(str(trajectory_id), int(action_index))] = reasoning
    return lookup


def first_present(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def record_reasoning(
    *,
    source: str,
    trajectory_id: str,
    task: str,
    action: Mapping[str, Any],
    action_index: int,
    lookup: Optional[Mapping[str, str]],
) -> Tuple[str, str]:
    if source == "none":
        return "", "none"
    if source == "template":
        return reasoning_for(task, action), "synthetic_template"
    if source == "jsonl":
        if lookup is None:
            raise ValueError("reasoning lookup is required when source=jsonl")
        text = lookup.get(reasoning_key(trajectory_id, action_index), "")
        return text, "jsonl" if text else "missing"
    raise ValueError(f"Unsupported reasoning source: {source}")


def build_windows(
    frame_names: List[str],
    actions: List[Mapping[str, Any]],
    *,
    key: str,
    task: str,
    frame_root_rel: str,
    max_window_frames: int,
    stride: int,
    reasoning_source: str,
    reasoning_lookup: Optional[Mapping[str, str]],
    require_reasoning: bool,
) -> Iterable[Dict[str, Any]]:
    for frame_pos in range(max_window_frames - 1, len(frame_names)):
        action_idx = frame_pos * stride
        if action_idx >= len(actions):
            break
        action = actions[action_idx]
        reasoning, reasoning_source_name = record_reasoning(
            source=reasoning_source,
            trajectory_id=key,
            task=task,
            action=action,
            action_index=action_idx,
            lookup=reasoning_lookup,
        )
        if require_reasoning and not reasoning:
            continue
        frames = [
            f"{frame_root_rel}/{frame_names[index]}"
            for index in range(frame_pos - max_window_frames + 1, frame_pos + 1)
        ]
        yield {
            "frames": frames,
            "task": task_text(task),
            "actions": [normalize_minerl_action(action, source=SOURCE)],
            "reasoning": reasoning,
            "metadata": {
                "source": SOURCE,
                "stage_objective": "frames_truncated_aot_goal_action_alignment",
                "trajectory_id": key,
                "task_name": task,
                "action_index": action_idx,
                "video_frame_index": action_idx,
                "frame_stride": stride,
                "reasoning_source": reasoning_source_name,
            },
        }


def split_keys(keys: List[str], val_ratio: float, seed: int) -> tuple[List[str], set[str]]:
    rng = random.Random(seed)
    shuffled = list(keys)
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio)) if shuffled else 0
    return shuffled, set(shuffled[:val_count])


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Optimus-2-MGOA into Stage3 JSONL manifests.")
    parser.add_argument("--root", required=True, help="Directory with action.tar.gz, video.tar.gz, and task map.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-count", type=int, default=100000)
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--max-window-frames", type=int, default=4)
    parser.add_argument("--max-frames-per-trajectory", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-existing-frames", action="store_true")
    parser.add_argument(
        "--reasoning-source",
        choices=("none", "template", "jsonl"),
        default="none",
        help=(
            "Reasoning provenance. Optimus-2-MGOA has task/action/frame labels but no native AoT explanations; "
            "'template' is synthetic and intended only for controlled ablations."
        ),
    )
    parser.add_argument(
        "--reasoning-jsonl",
        default=None,
        help=(
            "Optional JSONL with trajectory_id/key/video_id, action_index/frame_idx, and reasoning fields. "
            "Used only with --reasoning-source jsonl."
        ),
    )
    parser.add_argument(
        "--require-reasoning",
        action="store_true",
        help="Skip Stage3 windows that do not have non-empty reasoning from the selected source.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_root = Path(args.output_root)
    frames_root = output_root / "frames_optimus_mgoa_stage3"
    output_root.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)

    action_tar = root / "action.tar.gz"
    video_tar = root / "video.tar.gz"
    task_map_path = root / "task_description_map.json"
    for path in (action_tar, video_tar, task_map_path):
        if not path.exists():
            raise FileNotFoundError(path)

    task_map = load_task_map(task_map_path)
    reasoning_lookup = None
    if args.reasoning_source == "jsonl":
        if not args.reasoning_jsonl:
            raise ValueError("--reasoning-jsonl is required with --reasoning-source jsonl")
        reasoning_lookup = load_reasoning_jsonl(Path(args.reasoning_jsonl))
        if not reasoning_lookup:
            raise ValueError(f"No reasoning records found in {args.reasoning_jsonl}")
    elif args.reasoning_jsonl:
        raise ValueError("--reasoning-jsonl can only be used with --reasoning-source jsonl")

    ordered_keys, val_keys = split_keys(sorted(task_map), args.val_ratio, args.seed)
    if args.max_trajectories:
        ordered_keys = ordered_keys[: args.max_trajectories]
    selected_keys = set(ordered_keys)

    train_path = output_root / "train_stage3.jsonl"
    val_path = output_root / "val_stage3.jsonl"
    counts = {"train": 0, "val": 0, "trajectories": 0, "missing_video": 0}

    with train_path.open("w", encoding="utf-8") as train_handle, val_path.open("w", encoding="utf-8") as val_handle:
        with tempfile.TemporaryDirectory(prefix="optimus_mgoa_") as tmp:
            tmp_dir = Path(tmp)
            actions_dir = tmp_dir / "actions"
            action_keys = extract_action_pickles(action_tar, selected_keys, actions_dir)
            counts["missing_action"] = len(selected_keys - action_keys)
            remaining_keys = set(key for key in ordered_keys if key in action_keys)
            with tarfile.open(video_tar, "r|gz") as videos:
                for member in videos:
                    if not member.isfile() or not member.name.startswith("video/") or not member.name.endswith(".mp4"):
                        continue
                    key = Path(member.name).stem
                    if key not in remaining_keys:
                        continue

                    remaining_keys.remove(key)
                    if counts["train"] >= args.train_count and counts["val"] >= args.val_count:
                        break

                    trajectory_frame_dir = frames_root / key
                    frame_root_rel = f"frames_optimus_mgoa_stage3/{key}"
                    if args.skip_existing_frames and trajectory_frame_dir.exists():
                        frame_names = sorted(path.name for path in trajectory_frame_dir.glob("frame_*.jpg"))
                    else:
                        shutil.rmtree(tmp_dir / "video", ignore_errors=True)
                        extracted = videos.extractfile(member)
                        if extracted is None:
                            continue
                        video_path = tmp_dir / "video" / f"{key}.mp4"
                        video_path.parent.mkdir(parents=True, exist_ok=True)
                        with video_path.open("wb") as out:
                            shutil.copyfileobj(extracted, out, length=1024 * 1024)
                        frame_names = write_frames(
                            video_path,
                            trajectory_frame_dir,
                            stride=args.frame_stride,
                            max_frames=args.max_frames_per_trajectory,
                        )
                        shutil.rmtree(tmp_dir / "video", ignore_errors=True)

                    if len(frame_names) < args.max_window_frames:
                        continue

                    actions = load_actions_from_file(actions_dir, key)
                    task = task_map[key]
                    target = "val" if key in val_keys else "train"
                    for record in build_windows(
                        frame_names,
                        actions,
                        key=key,
                        task=task,
                        frame_root_rel=frame_root_rel,
                        max_window_frames=args.max_window_frames,
                        stride=args.frame_stride,
                        reasoning_source=args.reasoning_source,
                        reasoning_lookup=reasoning_lookup,
                        require_reasoning=args.require_reasoning,
                    ):
                        if target == "val":
                            if counts["val"] >= args.val_count:
                                break
                            val_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                            counts["val"] += 1
                        else:
                            if counts["train"] >= args.train_count:
                                break
                            train_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                            counts["train"] += 1

                    counts["trajectories"] += 1
                    if counts["trajectories"] % 25 == 0:
                        print(json.dumps(counts, ensure_ascii=False), flush=True)

            counts["missing_video"] = len(remaining_keys)

    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print(train_path)
    print(val_path)


if __name__ == "__main__":
    main()
