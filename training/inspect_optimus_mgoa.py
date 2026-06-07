"""Inspect Optimus-2-MGOA action/task metadata without downloading videos."""

from __future__ import annotations

import argparse
import json
import pickle
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any


def scalar(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1:
        return scalar(value[0])
    return value


def summarize_action(action: dict[str, Any]) -> dict[str, Any]:
    return {key: scalar(value) for key, value in action.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Optimus-2-MGOA action.tar.gz and task map.")
    parser.add_argument("--root", required=True, help="Directory containing action.tar.gz and task_description_map.json.")
    parser.add_argument("--sample-count", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root)
    task_map = json.loads((root / "task_description_map.json").read_text(encoding="utf-8"))
    print(f"tasks={len(task_map)} unique_tasks={len(set(task_map.values()))}")
    print("top_tasks=" + json.dumps(Counter(task_map.values()).most_common(20), ensure_ascii=False))

    with tarfile.open(root / "action.tar.gz", "r:gz") as tar:
        members = [member for member in tar.getmembers() if member.name.endswith(".pkl")]
        print(f"action_files={len(members)}")
        missing = [Path(member.name).stem for member in members if Path(member.name).stem not in task_map]
        print(f"missing_task_keys={len(missing)}")

        for member in members[: args.sample_count]:
            key = Path(member.name).stem
            handle = tar.extractfile(member)
            if handle is None:
                continue
            actions = pickle.load(handle)
            print(f"sample={key} task={task_map.get(key)} frames={len(actions)}")
            if actions:
                print(json.dumps(summarize_action(actions[0]), ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
