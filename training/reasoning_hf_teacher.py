"""Local Hugging Face VLM teacher for Action-of-Thought reasoning.

This is a fallback/alternative to vLLM endpoints. It loads one multimodal model
inside the job and writes the same reasoning JSONL artifact consumed by
`training.reasoning_annotation merge`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from training.image_refs import TessStage1ImageStore, load_image
from training.reasoning_annotation import (
    action_text,
    build_prompt,
    existing_ids,
    parse_json_response,
    record_id,
    select_frame_indices,
)
from training.dataset import MinecraftVLADataset


def fallback_reasoning_from_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    for fence in ("```json", "```"):
        cleaned = cleaned.replace(fence, "")
    cleaned = cleaned.strip()
    prefixes = ("reasoning:", "explanation:", "answer:")
    lowered = cleaned.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break
    return " ".join(cleaned.split())


def load_model(model_name: str, torch_dtype: str):
    dtype = getattr(torch, torch_dtype)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return model, processor


def generate_reasoning(
    *,
    model,
    processor,
    prompt: str,
    images,
    max_new_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
            + [{"type": "image", "image": image} for image in images],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )
    prompt_len = inputs["input_ids"].shape[1]
    output_ids = generated[:, prompt_len:]
    text_out = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    try:
        parsed = parse_json_response(text_out)
    except json.JSONDecodeError as exc:
        parsed = {
            "reasoning": fallback_reasoning_from_text(text_out),
            "confidence": 0.3,
            "tags": ["fallback_text_parse"],
            "parse_error": type(exc).__name__,
        }
    parsed["raw_text"] = text_out
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate reasoning with a local Hugging Face VLM.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--stage", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--tess-stage1-index", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-every", type=int, default=10)
    args = parser.parse_args()

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
    model, processor = load_model(args.model, args.torch_dtype)

    written = 0
    scanned = 0
    with output_path.open("a", encoding="utf-8") as out:
        for index in range(args.start, len(dataset)):
            if args.limit and scanned >= args.limit:
                break
            scanned += 1
            raw = dataset._read_record(index)
            rid = record_id(raw, index + 1)
            if rid in done:
                continue
            frames = raw.get("frames") or []
            images = [
                load_image(frames[i], store)
                for i in select_frame_indices(len(frames), args.max_images)
            ]
            prompt = build_prompt(args.stage, str(raw.get("task") or ""), action_text(raw))
            try:
                result = generate_reasoning(
                    model=model,
                    processor=processor,
                    prompt=prompt,
                    images=images,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
            except Exception as exc:  # noqa: BLE001
                result = {
                    "reasoning": "",
                    "confidence": 0.0,
                    "tags": ["generation_error"],
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
            artifact = {
                "record_id": rid,
                "line_number": index + 1,
                "stage": args.stage,
                "task": raw.get("task") or "",
                "reasoning": str(result.get("reasoning") or "").strip(),
                "confidence": result.get("confidence", 0.0),
                "tags": result.get("tags", []),
                "reasoning_source": "teacher_vlm",
                "selected_provider": "hf_local",
                "selected_model": args.model,
                "provider_outputs": [result],
                "metadata": raw.get("metadata"),
            }
            out.write(json.dumps(artifact, ensure_ascii=False, separators=(",", ":")) + "\n")
            out.flush()
            written += 1
            if written % args.report_every == 0:
                print(json.dumps({"written": written, "scanned": scanned, "last_record_id": rid}), flush=True)

    print(json.dumps({"written": written, "scanned": scanned, "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
