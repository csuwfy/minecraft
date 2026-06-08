"""Local Hugging Face VLM teacher for Action-of-Thought reasoning.

This is a fallback/alternative to vLLM endpoints. It loads one multimodal model
inside the job and writes the same reasoning JSONL artifact consumed by
`training.reasoning_annotation merge`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from training.image_refs import TessStage1ImageStore, load_image
from training.reasoning_annotation import (
    action_text,
    build_prompt,
    existing_ids,
    normalize_reasoning_text,
    parse_json_response,
    record_id,
    select_frame_indices,
)
from training.dataset import MinecraftVLADataset


def fallback_reasoning_from_text(text: str) -> str:
    return normalize_reasoning_text(text)


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


def generate_reasoning_batch(
    *,
    model,
    processor,
    prompts: Sequence[str],
    images_batch: Sequence[Sequence[Any]],
    max_new_tokens: int,
    temperature: float,
) -> List[Dict[str, Any]]:
    messages_batch = [
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
                + [{"type": "image", "image": image} for image in images],
            }
        ]
        for prompt, images in zip(prompts, images_batch)
    ]
    texts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]
    inputs = processor(text=texts, images=list(images_batch), padding=True, return_tensors="pt")
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
    decoded = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    results: List[Dict[str, Any]] = []
    for text_out in decoded:
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
        results.append(parsed)
    return results


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def model_safe_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def output_for_chunk(
    *,
    output_dir: Path,
    stage: int,
    split: str,
    model_name: str,
    start: int,
    limit: int,
) -> Path:
    return output_dir / (
        f"reasoning_stage{stage}_{split}_{model_safe_name(model_name)}_"
        f"start{start:09d}_limit{limit}.jsonl"
    )


def artifact_from_result(
    *,
    rid: str,
    line_number: int,
    stage: int,
    task: str,
    result: Mapping[str, Any],
    model_name: str,
    metadata: Any,
) -> Dict[str, Any]:
    return {
        "record_id": rid,
        "line_number": line_number,
        "stage": stage,
        "task": task,
        "reasoning": str(result.get("reasoning") or "").strip(),
        "confidence": result.get("confidence", 0.0),
        "tags": result.get("tags", []),
        "reasoning_source": "teacher_vlm",
        "selected_provider": "hf_local",
        "selected_model": model_name,
        "provider_outputs": [result],
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate reasoning with a local Hugging Face VLM.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--stage", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--chunk-output-dir",
        default=None,
        help="Write one JSONL per aligned chunk inside this directory, preserving merge coverage names.",
    )
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tess-stage1-index", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--report-every", type=int, default=10)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if not args.output and not args.chunk_output_dir:
        raise SystemExit("Set either --output or --chunk-output-dir.")
    if args.chunk_output_dir and args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive when --chunk-output-dir is set.")

    dataset = MinecraftVLADataset(
        manifest_path=args.manifest,
        image_root=args.image_root,
        stage=args.stage,
        max_frames=args.max_frames,
        tess_stage1_index=args.tess_stage1_index,
    )
    store = TessStage1ImageStore(args.tess_stage1_index) if args.tess_stage1_index else None
    model, processor = load_model(args.model, args.torch_dtype)

    end_index = len(dataset)
    if args.limit:
        end_index = min(end_index, args.start + args.limit)

    def build_example(index: int) -> Dict[str, Any]:
        raw = dataset._read_record(index)
        rid = record_id(raw, index + 1)
        frames = raw.get("frames") or []
        images = [
            load_image(frames[i], store)
            for i in select_frame_indices(len(frames), args.max_images)
        ]
        task = str(raw.get("task") or "")
        prompt = build_prompt(args.stage, task, action_text(raw))
        return {
            "index": index,
            "raw": raw,
            "record_id": rid,
            "images": images,
            "task": task,
            "prompt": prompt,
        }

    def generate_batch_with_fallback(batch: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        prompts = [str(item["prompt"]) for item in batch]
        images_batch = [item["images"] for item in batch]
        try:
            return generate_reasoning_batch(
                model=model,
                processor=processor,
                prompts=prompts,
                images_batch=images_batch,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        except Exception as exc:  # noqa: BLE001
            if len(batch) == 1:
                return [
                    {
                        "reasoning": "",
                        "confidence": 0.0,
                        "tags": ["generation_error"],
                        "error": type(exc).__name__,
                        "message": str(exc),
                    }
                ]
            results: List[Dict[str, Any]] = []
            for item in batch:
                try:
                    results.append(
                        generate_reasoning(
                            model=model,
                            processor=processor,
                            prompt=str(item["prompt"]),
                            images=item["images"],
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                        )
                    )
                except Exception as inner_exc:  # noqa: BLE001
                    results.append(
                        {
                            "reasoning": "",
                            "confidence": 0.0,
                            "tags": ["generation_error"],
                            "error": type(inner_exc).__name__,
                            "message": str(inner_exc),
                        }
                    )
            return results

    written = 0
    scanned = 0

    if args.chunk_output_dir:
        output_dir = Path(args.chunk_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_size = args.chunk_size
        chunk_start = (args.start // chunk_size) * chunk_size
        if chunk_start != args.start:
            raise SystemExit(f"--start must align to --chunk-size in chunk-output mode: {args.start} vs {chunk_size}")
        index = args.start
        while index < end_index:
            chunk_start = index
            chunk_end = min(chunk_start + chunk_size, len(dataset), end_index)
            chunk_limit = chunk_end - chunk_start
            output_path = output_for_chunk(
                output_dir=output_dir,
                stage=args.stage,
                split=args.split,
                model_name=args.model,
                start=chunk_start,
                limit=chunk_limit,
            )
            existing_rows = count_jsonl_rows(output_path)
            if existing_rows == chunk_limit:
                print(
                    json.dumps(
                        {
                            "skip_complete_chunk": str(output_path),
                            "start": chunk_start,
                            "limit": chunk_limit,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                index = chunk_end
                continue
            if existing_rows > chunk_limit:
                raise SystemExit(
                    f"chunk_row_overflow output={output_path} rows={existing_rows} limit={chunk_limit}"
                )
            done = existing_ids(output_path) if args.resume and existing_rows else set()
            batch: List[Dict[str, Any]] = []
            with output_path.open("a", encoding="utf-8") as out:
                for record_index in range(chunk_start, chunk_end):
                    scanned += 1
                    example = build_example(record_index)
                    if example["record_id"] in done:
                        continue
                    batch.append(example)
                    if len(batch) < args.batch_size:
                        continue
                    results = generate_batch_with_fallback(batch)
                    for item, result in zip(batch, results):
                        artifact = artifact_from_result(
                            rid=str(item["record_id"]),
                            line_number=int(item["index"]) + 1,
                            stage=args.stage,
                            task=str(item["task"]),
                            result=result,
                            model_name=args.model,
                            metadata=item["raw"].get("metadata"),
                        )
                        out.write(json.dumps(artifact, ensure_ascii=False, separators=(",", ":")) + "\n")
                        written += 1
                    out.flush()
                    if written % args.report_every == 0:
                        print(
                            json.dumps(
                                {
                                    "written": written,
                                    "scanned": scanned,
                                    "last_record_id": batch[-1]["record_id"],
                                    "output": str(output_path),
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                    batch = []
                if batch:
                    results = generate_batch_with_fallback(batch)
                    for item, result in zip(batch, results):
                        artifact = artifact_from_result(
                            rid=str(item["record_id"]),
                            line_number=int(item["index"]) + 1,
                            stage=args.stage,
                            task=str(item["task"]),
                            result=result,
                            model_name=args.model,
                            metadata=item["raw"].get("metadata"),
                        )
                        out.write(json.dumps(artifact, ensure_ascii=False, separators=(",", ":")) + "\n")
                        written += 1
                    out.flush()
            final_rows = count_jsonl_rows(output_path)
            print(
                json.dumps(
                    {
                        "chunk_done": str(output_path),
                        "start": chunk_start,
                        "limit": chunk_limit,
                        "rows": final_rows,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if final_rows != chunk_limit:
                raise SystemExit(f"chunk_incomplete output={output_path} rows={final_rows} limit={chunk_limit}")
            index = chunk_end
        print(
            json.dumps(
                {
                    "written": written,
                    "scanned": scanned,
                    "chunk_output_dir": str(output_dir),
                    "start": args.start,
                    "end": end_index,
                },
                ensure_ascii=False,
            )
        )
        return

    output_path = Path(str(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = existing_ids(output_path) if args.resume else set()
    batch: List[Dict[str, Any]] = []
    with output_path.open("a", encoding="utf-8") as out:
        for index in range(args.start, end_index):
            scanned += 1
            example = build_example(index)
            if example["record_id"] in done:
                continue
            batch.append(example)
            if len(batch) < args.batch_size:
                continue
            results = generate_batch_with_fallback(batch)
            for item, result in zip(batch, results):
                artifact = artifact_from_result(
                    rid=str(item["record_id"]),
                    line_number=int(item["index"]) + 1,
                    stage=args.stage,
                    task=str(item["task"]),
                    result=result,
                    model_name=args.model,
                    metadata=item["raw"].get("metadata"),
                )
                out.write(json.dumps(artifact, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1
            out.flush()
            if written % args.report_every == 0:
                print(json.dumps({"written": written, "scanned": scanned, "last_record_id": batch[-1]["record_id"]}), flush=True)
            batch = []
        if batch:
            results = generate_batch_with_fallback(batch)
            for item, result in zip(batch, results):
                artifact = artifact_from_result(
                    rid=str(item["record_id"]),
                    line_number=int(item["index"]) + 1,
                    stage=args.stage,
                    task=str(item["task"]),
                    result=result,
                    model_name=args.model,
                    metadata=item["raw"].get("metadata"),
                )
                out.write(json.dumps(artifact, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1
            out.flush()

    print(json.dumps({"written": written, "scanned": scanned, "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
