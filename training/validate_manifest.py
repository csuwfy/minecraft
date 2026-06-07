"""Validate and preview a Minecraft VLA JSONL manifest."""

from __future__ import annotations

import argparse
import json

from training.dataset import MinecraftVLADataset
from training.image_refs import TessStage1ImageStore, load_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Minecraft VLA JSONL manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=3)
    parser.add_argument("--preview", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="Only touch the first N records for a quick check.")
    parser.add_argument("--tess-stage1-index", default=None, help="Parquet index used by TESS Stage1 frame references.")
    parser.add_argument("--load-images", action="store_true", help="Open every frame in the checked records.")
    parser.add_argument("--require-reasoning", action="store_true", help="Reject records with empty AoT reasoning.")
    args = parser.parse_args()

    dataset = MinecraftVLADataset(
        manifest_path=args.manifest,
        image_root=args.image_root,
        stage=args.stage,
        max_frames=args.max_frames,
        tess_stage1_index=args.tess_stage1_index,
        require_reasoning=args.require_reasoning,
    )
    print(f"records={len(dataset)}")
    touch_count = min(args.limit or args.preview, len(dataset))
    store = TessStage1ImageStore(args.tess_stage1_index) if args.load_images and args.tess_stage1_index else None
    loaded_images = 0
    for index in range(touch_count):
        item = dataset[index]
        if args.load_images:
            for frame in item["frames"]:
                image = load_image(frame, store)
                image.load()
                loaded_images += 1
        if index < args.preview:
            print(json.dumps(item["messages"], ensure_ascii=False, indent=2)[:4000])
    if args.load_images:
        print(f"loaded_images={loaded_images}")


if __name__ == "__main__":
    main()
