"""Generate and merge teacher reasoning for Minecraft VLA manifests.

The public Minecraft datasets used by this scaffold do not include the
CombatVLA paper's Action-of-Thought explanations. This module keeps teacher
reasoning as a separate auditable artifact before it is merged into manifests.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import requests
from PIL import Image

from training.dataset import MinecraftVLADataset
from training.image_refs import TessStage1ImageStore, load_image


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    api_url: str = ""
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 320
    timeout: int = 120
    mock: bool = False


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def provider_from_config(record: Mapping[str, Any]) -> Provider:
    api_key = str(record.get("api_key") or "")
    api_key_env = str(record.get("api_key_env") or "")
    if api_key_env:
        api_key = os.environ.get(api_key_env, api_key)
    return Provider(
        name=str(record["name"]),
        model=str(record.get("model") or record["name"]),
        api_url=str(record.get("api_url") or "").rstrip("/"),
        api_key=api_key,
        temperature=float(record.get("temperature", 0.2)),
        max_tokens=int(record.get("max_tokens", 320)),
        timeout=int(record.get("timeout", 120)),
        mock=bool(record.get("mock", False)),
    )


def record_id(record: Mapping[str, Any], line_number: int) -> str:
    metadata = record.get("metadata") or {}
    if isinstance(metadata, Mapping):
        trajectory_id = metadata.get("trajectory_id") or metadata.get("video_id") or metadata.get("id")
        action_index = first_present(metadata, "action_index", "frame_idx", "video_frame_index")
        if trajectory_id is not None and action_index is not None:
            return f"{trajectory_id}:{int(action_index)}"
        if trajectory_id is not None:
            return f"{trajectory_id}:line{line_number}"
    return f"line:{line_number}"


def first_present(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def action_text(record: Mapping[str, Any]) -> str:
    actions = record.get("actions")
    if actions is None and "action" in record:
        actions = [record["action"]]
    return json.dumps({"actions": actions or []}, ensure_ascii=False, separators=(",", ":"))


def select_frame_indices(frame_count: int, max_images: int) -> List[int]:
    if frame_count <= 0:
        return []
    if max_images <= 0 or frame_count <= max_images:
        return list(range(frame_count))
    if max_images == 1:
        return [frame_count - 1]
    indices = []
    for i in range(max_images):
        indices.append(round(i * (frame_count - 1) / (max_images - 1)))
    return sorted(set(indices))


def image_to_data_url(image: Image.Image, max_side: int, quality: int) -> str:
    rgb = image.convert("RGB")
    if max_side > 0:
        width, height = rgb.size
        scale = min(1.0, max_side / max(width, height))
        if scale < 1.0:
            rgb = rgb.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    buffer = BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_prompt(stage: int, task: str, actions: str) -> str:
    if stage in (1, 2):
        target = "The final training target will place reasoning before the action JSON."
    else:
        target = "The final training target will place action JSON before <TRUNC>, then the reasoning."
    return (
        "You are labeling Action-of-Thought reasoning for a Minecraft vision-language-action dataset.\n"
        "Given the visual frames, task, and gold next action, write a concise explanation for why the gold action "
        "is appropriate. Ground the explanation in visible state, task progress, and action semantics. Do not invent "
        "objects or game state that cannot be inferred from the frames. Keep it one or two sentences.\n"
        f"{target}\n"
        "Return strict JSON only with keys: reasoning, confidence, tags.\n"
        f"Stage: {stage}\n"
        f"Task: {task or 'Play Minecraft.'}\n"
        f"Gold action JSON: {actions}"
    )


def parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = JSON_RE.search(text)
        if not match:
            raise
        return json.loads(match.group(0))


def call_provider(provider: Provider, prompt: str, image_urls: List[str]) -> Dict[str, Any]:
    if provider.mock:
        return {
            "reasoning": "The selected gold action follows the current Minecraft task and recent visual context.",
            "confidence": 0.1,
            "tags": ["mock"],
            "raw_text": "",
        }
    if not provider.api_url:
        raise ValueError(f"Provider {provider.name} requires api_url.")
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": image_url}} for image_url in image_urls)
    payload = {
        "model": provider.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": provider.temperature,
        "max_tokens": provider.max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if provider.api_key:
        headers["Authorization"] = f"Bearer {provider.api_key}"
    response = requests.post(
        f"{provider.api_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=provider.timeout,
    )
    response.raise_for_status()
    data = response.json()
    text = data["choices"][0]["message"]["content"]
    parsed = parse_json_response(text)
    parsed["raw_text"] = text
    return parsed


def choose_reasoning(outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = []
    for output in outputs:
        reasoning = str(output.get("reasoning") or "").strip()
        if not reasoning:
            continue
        confidence = output.get("confidence", 0.0)
        try:
            score = float(confidence)
        except (TypeError, ValueError):
            score = 0.0
        valid.append((score, len(reasoning), output))
    if not valid:
        return {"reasoning": "", "confidence": 0.0, "tags": ["no_valid_reasoning"]}
    valid.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return valid[0][2]


def existing_ids(output_path: Path) -> set[str]:
    ids: set[str] = set()
    if not output_path.exists():
        return ids
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            rid = record.get("record_id")
            if rid:
                ids.add(str(rid))
    return ids


def annotate(args: argparse.Namespace) -> None:
    config = load_json(args.config)
    providers = [provider_from_config(item) for item in config.get("providers", [])]
    if not providers:
        raise ValueError("Reasoning config requires at least one provider.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = existing_ids(output_path) if args.resume else set()

    dataset = MinecraftVLADataset(
        manifest_path=args.manifest,
        image_root=args.image_root,
        stage=args.stage,
        max_frames=args.max_frames,
        tess_stage1_index=args.tess_stage1_index,
    )
    store = TessStage1ImageStore(args.tess_stage1_index) if args.tess_stage1_index else None
    max_images = int(config.get("max_images", args.max_images))
    max_side = int(config.get("max_image_side", args.max_image_side))
    quality = int(config.get("jpeg_quality", args.jpeg_quality))
    sleep_seconds = float(config.get("sleep_seconds", args.sleep_seconds))

    written = 0
    scanned = 0
    with output_path.open("a", encoding="utf-8") as out:
        for index in range(args.start, len(dataset)):
            if args.limit and scanned >= args.limit:
                break
            scanned += 1
            raw = dataset._read_record(index)  # Reuses manifest offsets; avoids full JSONL scans.
            rid = record_id(raw, index + 1)
            if rid in done:
                continue
            frames = raw.get("frames") or []
            selected = select_frame_indices(len(frames), max_images)
            image_urls = [
                image_to_data_url(load_image(frames[i], store), max_side=max_side, quality=quality)
                for i in selected
            ]
            prompt = build_prompt(args.stage, str(raw.get("task") or ""), action_text(raw))
            outputs = []
            for provider in providers:
                try:
                    result = call_provider(provider, prompt, image_urls)
                    result["provider"] = provider.name
                    result["model"] = provider.model
                    outputs.append(result)
                except Exception as exc:  # noqa: BLE001 - record provider failures for audit.
                    outputs.append(
                        {
                            "provider": provider.name,
                            "model": provider.model,
                            "error": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
            chosen = choose_reasoning(outputs)
            artifact = {
                "record_id": rid,
                "line_number": index + 1,
                "stage": args.stage,
                "task": raw.get("task") or "",
                "reasoning": str(chosen.get("reasoning") or "").strip(),
                "confidence": chosen.get("confidence", 0.0),
                "tags": chosen.get("tags", []),
                "reasoning_source": "teacher_vlm",
                "selected_provider": chosen.get("provider"),
                "selected_model": chosen.get("model"),
                "provider_outputs": outputs,
                "metadata": raw.get("metadata"),
            }
            out.write(json.dumps(artifact, ensure_ascii=False, separators=(",", ":")) + "\n")
            out.flush()
            written += 1
            if written % args.report_every == 0:
                print(json.dumps({"written": written, "scanned": scanned, "last_record_id": rid}), flush=True)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    print(json.dumps({"written": written, "scanned": scanned, "output": str(output_path)}, ensure_ascii=False))


def load_reasoning(path: Path) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            rid = str(record.get("record_id") or "")
            reasoning = str(record.get("reasoning") or "").strip()
            if rid and reasoning:
                result[rid] = record
    return result


def iter_manifest(path: Path) -> Iterable[tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            yield line_number, json.loads(line)


def merge(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reasoning = load_reasoning(Path(args.reasoning_jsonl))
    merged = 0
    total = 0
    with output_path.open("w", encoding="utf-8") as out:
        for line_number, record in iter_manifest(manifest_path):
            total += 1
            rid = record_id(record, line_number)
            teacher = reasoning.get(rid)
            if teacher is not None:
                record["reasoning"] = str(teacher.get("reasoning") or "").strip()
                metadata = dict(record.get("metadata") or {})
                metadata["reasoning_source"] = teacher.get("reasoning_source", "teacher_vlm")
                metadata["reasoning_record_id"] = rid
                metadata["reasoning_model"] = teacher.get("selected_model")
                metadata["reasoning_provider"] = teacher.get("selected_provider")
                metadata["reasoning_confidence"] = teacher.get("confidence")
                record["metadata"] = metadata
                merged += 1
            elif args.require_reasoning:
                continue
            out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"total": total, "merged": merged, "output": str(output_path)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher reasoning annotation utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ann = subparsers.add_parser("annotate", help="Generate teacher reasoning JSONL.")
    ann.add_argument("--manifest", required=True)
    ann.add_argument("--image-root", required=True)
    ann.add_argument("--stage", type=int, required=True, choices=(1, 2, 3))
    ann.add_argument("--config", required=True)
    ann.add_argument("--output", required=True)
    ann.add_argument("--max-frames", type=int, default=4)
    ann.add_argument("--max-images", type=int, default=4)
    ann.add_argument("--max-image-side", type=int, default=448)
    ann.add_argument("--jpeg-quality", type=int, default=85)
    ann.add_argument("--tess-stage1-index", default=None)
    ann.add_argument("--start", type=int, default=0)
    ann.add_argument("--limit", type=int, default=0)
    ann.add_argument("--resume", action="store_true")
    ann.add_argument("--sleep-seconds", type=float, default=0.0)
    ann.add_argument("--report-every", type=int, default=50)
    ann.set_defaults(func=annotate)

    mrg = subparsers.add_parser("merge", help="Merge teacher reasoning JSONL into a manifest.")
    mrg.add_argument("--manifest", required=True)
    mrg.add_argument("--reasoning-jsonl", required=True)
    mrg.add_argument("--output", required=True)
    mrg.add_argument("--require-reasoning", action="store_true")
    mrg.set_defaults(func=merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
