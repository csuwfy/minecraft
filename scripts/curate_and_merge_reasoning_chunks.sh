#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/minecraft_full}"
REASONING_ROOT="${REASONING_ROOT:-$PROJECT_ROOT/data/reasoning}"
MERGED_ROOT="${MERGED_ROOT:-$PROJECT_ROOT/data/minecraft_reasoning}"
STAGE="${STAGE:?Set STAGE to 1, 2, or 3.}"
SPLIT="${SPLIT:-train}"
MODEL_GLOBS="${MODEL_GLOBS:-Qwen_Qwen2.5-VL-7B-Instruct Qwen_Qwen2.5-VL-3B-Instruct OpenGVLab_InternVL3-2B HuggingFaceTB_SmolVLM2-2.2B-Instruct}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${ENV:-}" ]]; then
    PYTHON_BIN="$ENV/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

mkdir -p "$REASONING_ROOT" "$MERGED_ROOT"

MANIFEST="$DATA_ROOT/${SPLIT}_stage${STAGE}.jsonl"
CURATED="$REASONING_ROOT/reasoning_stage${STAGE}_${SPLIT}_curated.jsonl"
REJECTED="$REASONING_ROOT/reasoning_stage${STAGE}_${SPLIT}_rejected.jsonl"
MERGED="$MERGED_ROOT/${SPLIT}_stage${STAGE}.jsonl"
REPORT="$MERGED_ROOT/${SPLIT}_stage${STAGE}_merge_report.json"

if [[ ! -s "$MANIFEST" ]]; then
  echo "missing_source_manifest=$MANIFEST" >&2
  exit 2
fi

reasoning_files=()
read -r -a model_globs <<< "$MODEL_GLOBS"
for model_glob in "${model_globs[@]}"; do
  while IFS= read -r file; do
    reasoning_files+=("$file")
  done < <(
    find "$REASONING_ROOT" -maxdepth 1 -type f \
      -name "reasoning_stage${STAGE}_${SPLIT}_${model_glob}_start*_limit*.jsonl" \
      -print | sort
  )
done

if [[ "${#reasoning_files[@]}" -eq 0 ]]; then
  echo "no_reasoning_chunks stage=$STAGE split=$SPLIT root=$REASONING_ROOT" >&2
  exit 2
fi

"$PYTHON_BIN" - <<'PY' "$MANIFEST" "${reasoning_files[@]}"
import re
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
files = [Path(path) for path in sys.argv[2:]]
pattern = re.compile(r"_start(?P<start>\d+)_limit(?P<limit>\d+)\.jsonl$")

source_count = 0
with manifest.open("rb") as handle:
    for line in handle:
        if line.strip():
            source_count += 1

ranges = []
for path in files:
    match = pattern.search(path.name)
    if not match:
        raise SystemExit(f"bad_reasoning_chunk_name={path}")
    start = int(match.group("start"))
    limit = int(match.group("limit"))
    with path.open("rb") as handle:
        rows = sum(1 for line in handle if line.strip())
    if rows != limit:
        raise SystemExit(f"chunk_row_mismatch file={path} limit={limit} rows={rows}")
    ranges.append((start, start + limit, path))

ranges.sort()
cursor = 0
for start, end, path in ranges:
    if start != cursor:
        raise SystemExit(f"chunk_coverage_gap expected_start={cursor} actual_start={start} file={path}")
    cursor = end

if cursor != source_count:
    raise SystemExit(f"partial_chunk_coverage covered={cursor} source_records={source_count}")

print(f"chunk_coverage_ok files={len(files)} source_records={source_count}")
PY

printf 'reasoning_chunk_count=%s\n' "${#reasoning_files[@]}"
printf 'reasoning_chunk=%s\n' "${reasoning_files[@]}"

"$PYTHON_BIN" -m training.reasoning_annotation curate-stream \
  --reasoning-jsonl "${reasoning_files[@]}" \
  --output "$CURATED" \
  --rejected-output "$REJECTED" \
  --preferred-model Qwen2.5-VL-7B-Instruct \
  --preferred-model Qwen2.5-VL-3B-Instruct \
  --preferred-model InternVL3-2B \
  --preferred-model SmolVLM2-2.2B-Instruct \
  --examples 20

"$PYTHON_BIN" -m training.reasoning_annotation merge-stream \
  --manifest "$MANIFEST" \
  --reasoning-jsonl "$CURATED" \
  --output "$MERGED" \
  --require-reasoning > "$REPORT"

"$PYTHON_BIN" - <<'PY' "$MANIFEST" "$MERGED" "$REPORT"
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
merged = Path(sys.argv[2])
report = Path(sys.argv[3])

def count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())

source_count = count(source)
merged_count = count(merged)
payload = json.loads(report.read_text(encoding="utf-8").strip().splitlines()[-1])
payload.update({"source_records": source_count, "merged_records": merged_count})
report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
if source_count != merged_count:
    raise SystemExit(
        f"partial_reasoning_merge source_records={source_count} merged_records={merged_count}"
    )
PY

echo "curated_reasoning=$CURATED"
echo "rejected_reasoning=$REJECTED"
echo "merged_manifest=$MERGED"
echo "merge_report=$REPORT"
