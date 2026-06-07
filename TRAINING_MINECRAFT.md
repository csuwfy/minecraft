# Minecraft Three-Stage VLA Training

This repository originally contains the CombatVLA action-execution framework.
The `training/` package is a public-data training scaffold for Minecraft. It is
organized to follow the CombatVLA paper's three-stage curriculum as closely as
the available open datasets allow.

## Paper Contract

The paper describes:

- A 3B VLA trained with Action-of-Thought (AoT).
- Full-parameter supervised fine-tuning, with the vision encoder frozen and the
  language model fine-tuned.
- Three progressive stages:
  - Stage 1: coarse video-level AoT.
  - Stage 2: fine frame-level AoT.
  - Stage 3: frame-level truncated AoT.
- Stage 1 and Stage 2 target format: `[explanation] [action]`.
- Stage 3 target format: `[action] <TRUNC> [explanation]`.
- Stage 1 uses `n = 20` frames and runs for `3` epochs.
- Stage 2 uses `k = 4` previous frames and runs for `1` epoch.
- Stage 3 runs for `3` epochs.
- Learning rate `1e-5`, batch size `1`, and a 95/5 train/validation split.
- The paper's full action-weighted objective combines language modeling,
  action-alignment loss, and modality contrastive loss.

What this scaffold implements:

- Qwen2.5-VL-3B-Instruct full-parameter SFT configs.
- Frozen vision parameters via `freeze_parameter_prefixes: ["visual"]`.
- The paper's stage counts, frame counts, LR, and batch size.
- Stage-specific target ordering, including `<TRUNC>` only for Stage 3.
- A transparent action-token weighted causal-LM loss.

What this scaffold cannot exactly reproduce from the paper:

- The private Black Myth: Wukong/Sekiro data and action tracker labels.
- The private action priority matcher, action-alignment loss, and modality
  contrastive loss. `training.trainer.ActionWeightedTrainer` is a practical
  approximation, not a full reproduction of those private terms.
- Human-written CombatVLA AoT explanations. Public Minecraft data has weaker or
  missing reasoning labels, so the converters never pretend synthetic text is
  the original paper annotation.

## Open Minecraft Data Plan

The three stages do not have to come from one dataset. In the paper they are a
curriculum over input/target format and action granularity. For public
Minecraft data, the cleanest mapping is:

| Stage | Main Dataset | Public Size | Why |
| --- | --- | ---: | --- |
| Stage 1 | `TESS-Computer/minecraft-vla-stage1` | 15,150,000 rows, 702 GB | Coarse frame/action pretraining, Lumine action chunks. |
| Stage 2 | `TESS-Computer/minecraft-vla-stage2` | 512,266 rows, 26 GB | Instruction-conditioned frame/action data, closest public fit for VLA SFT. |
| Stage 3 | `iLearn-Lab/Optimus-2-MGOA` | estimated 10,006,533 rows, 138 GB | Goal-observation-action trajectories with action/video alignment. |
| Supplement | `CraftJarvis/minecraft-vla-sft` | 106 GB | Useful only after keeping its action-token space separate or explicitly normalizing it. |

Sizes above are from the Hugging Face dataset file tree as checked on
2026-06-07.

Dataset URLs:

- Stage 1: https://huggingface.co/datasets/TESS-Computer/minecraft-vla-stage1
- Stage 2: https://huggingface.co/datasets/TESS-Computer/minecraft-vla-stage2
- Stage 3: https://huggingface.co/datasets/iLearn-Lab/Optimus-2-MGOA
- Optional supplement: https://huggingface.co/datasets/CraftJarvis/minecraft-vla-sft

## Repository Upload Policy

Do not upload datasets, converted frames, model checkpoints, caches, or
VideoSubFinder binaries to GitHub. They are intentionally ignored by
`.gitignore`:

- `data/`
- `datasets/`
- `cache/`
- `outputs/`
- `checkpoints/`
- `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `*.ckpt`
- `res/tool/subfinder/*` except the dummy `test.srt` and README

GitHub should contain only code, configs, scripts, and documentation. Download
large artifacts again on the Linux training machine.

## Linux L20 Environment

Recommended setup on the rented Linux machine:

```bash
git clone https://github.com/csuwfy/minecraft.git
cd minecraft
bash scripts/setup_l20_env.sh
conda activate combatvla-train
```

If you need DeepSpeed for memory pressure:

```bash
pip install -r requirements-deepspeed.txt
```

For Qwen2.5-VL, the training requirements include `transformers`,
`accelerate`, `qwen-vl-utils[decord]`, `datasets`, `pyarrow`, `opencv`, `peft`,
and Hugging Face download tools. Install the CUDA-matched PyTorch wheel before
`requirements-training.txt`.

## Download Data

Full download:

```bash
export PROJECT_ROOT=$PWD
export DATASETS_DIR=$PROJECT_ROOT/datasets
export HF_HOME=$PROJECT_ROOT/cache/huggingface
bash scripts/download_minecraft_datasets.sh
```

This downloads:

- `datasets/tess_stage1`
- `datasets/tess_stage2`
- `datasets/optimus2_mgoa/action.tar.gz`
- `datasets/optimus2_mgoa/task_description_map.json`
- `datasets/optimus2_mgoa/video.tar.gz.part.*`

The script assembles Optimus video parts into:

```text
datasets/optimus2_mgoa/video.tar.gz
```

To also download CraftJarvis:

```bash
DOWNLOAD_CRAFTJARVIS=1 bash scripts/download_minecraft_datasets.sh
```

## Convert Data

Full conversion:

```bash
export PROJECT_ROOT=$PWD
export DATASETS_DIR=$PROJECT_ROOT/datasets
export DATA_ROOT=$PROJECT_ROOT/data/minecraft_full
bash scripts/convert_minecraft_datasets.sh
```

Expected outputs:

```text
data/minecraft_full/train_stage1.jsonl
data/minecraft_full/val_stage1.jsonl
data/minecraft_full/tess_stage1_parquet_index.json
data/minecraft_full/train_stage2.jsonl
data/minecraft_full/val_stage2.jsonl
data/minecraft_full/frames_tess_stage2/
data/minecraft_full/train_stage3.jsonl
data/minecraft_full/val_stage3.jsonl
data/minecraft_full/frames_optimus_mgoa_stage3/
```

Stage 1 conversion uses parquet frame references by default. It does not copy
702 GB of images into another frame directory:

```bash
python -m training.convert_tess_stage1 \
  --data-dir datasets/tess_stage1 \
  --output-root data/minecraft_full \
  --train-count 100000000 \
  --val-count 10000 \
  --val-every 20 \
  --max-frames 20 \
  --task "Play Minecraft." \
  --reference-parquet-frames
```

Stage 2 conversion writes actual JPEG frames because the dataset is much
smaller and the frame/action/instruction rows are direct:

```bash
python -m training.convert_tess_stage2 \
  --data-dir datasets/tess_stage2 \
  --output-root data/minecraft_full \
  --train-count 100000000 \
  --val-count 10000 \
  --max-frames 4
```

Stage 3 conversion reads Optimus `task_description_map.json`, `action.tar.gz`,
and `video.tar.gz`, then builds goal/action/frame windows. Its reasoning text is
deterministic and explicitly derived from the goal and selected action; it is
not human-written AoT:

```bash
python -m training.convert_optimus_mgoa_stage3 \
  --root datasets/optimus2_mgoa \
  --output-root data/minecraft_full \
  --train-count 100000000 \
  --val-count 10000 \
  --frame-stride 4 \
  --max-window-frames 4 \
  --skip-existing-frames
```

For a smoke conversion, override counts:

```bash
STAGE1_TRAIN_COUNT=1000 STAGE1_VAL_COUNT=100 \
STAGE2_TRAIN_COUNT=1000 STAGE2_VAL_COUNT=100 \
STAGE3_TRAIN_COUNT=1000 STAGE3_VAL_COUNT=100 \
bash scripts/convert_minecraft_datasets.sh
```

## Validate Manifests

The conversion script already validates a few records and opens images. Manual
checks:

```bash
python -m training.validate_manifest \
  --manifest data/minecraft_full/train_stage1.jsonl \
  --image-root data/minecraft_full \
  --stage 1 \
  --max-frames 20 \
  --tess-stage1-index data/minecraft_full/tess_stage1_parquet_index.json \
  --limit 8 \
  --load-images

python -m training.validate_manifest \
  --manifest data/minecraft_full/train_stage2.jsonl \
  --image-root data/minecraft_full \
  --stage 2 \
  --max-frames 4 \
  --limit 8 \
  --load-images

python -m training.validate_manifest \
  --manifest data/minecraft_full/train_stage3.jsonl \
  --image-root data/minecraft_full \
  --stage 3 \
  --max-frames 4 \
  --limit 8 \
  --load-images
```

## Train Three Stages

Paper-style configs:

- `configs/training/paper_qwen25vl3b_stage1_full_sft.json`
- `configs/training/paper_qwen25vl3b_stage2_full_sft.json`
- `configs/training/paper_qwen25vl3b_stage3_full_sft.json`

They are chained:

- Stage 1 starts from `Qwen/Qwen2.5-VL-3B-Instruct`.
- Stage 2 starts from `outputs/paper-qwen25vl3b-stage1/final`.
- Stage 3 starts from `outputs/paper-qwen25vl3b-stage2/final`.

Run:

```bash
bash scripts/train_three_stage.sh
```

If a single L20 is too tight for Stage 1's 20-frame full-parameter run, keep the
paper config and add DeepSpeed CPU offload at runtime:

```bash
pip install -r requirements-deepspeed.txt
DEEPSPEED_CONFIG=configs/deepspeed/zero3_cpu_offload.json \
bash scripts/train_three_stage.sh
```

For multi-GPU L20 rental, prefer ZeRO-2:

```bash
pip install -r requirements-deepspeed.txt
DEEPSPEED_CONFIG=configs/deepspeed/zero2.json \
NUM_PROCESSES=2 \
bash scripts/train_three_stage.sh
```

Increase `NUM_PROCESSES` to the number of visible GPUs.

## Optional CraftJarvis Supplement

CraftJarvis uses reserved action tokens, not the Lumine action chunks used by
TESS. Do not directly concatenate it into TESS manifests unless you intentionally
standardize the action space.

Convert it as separate supplement manifests:

```bash
python -m training.convert_craftjarvis_vla \
  --data-dir datasets/craftjarvis_minecraft_vla_sft \
  --output-root data/minecraft_full \
  --train-count 100000 \
  --val-count 1000 \
  --stage23-frames 4
```

Outputs:

```text
train_stage2_craftjarvis.jsonl
val_stage2_craftjarvis.jsonl
```

## Windows Runtime Tool

The original game-control framework needs VideoSubFinder for subtitle/OCR
processing. Do not upload its extracted binaries. Download it from:

https://sourceforge.net/projects/videosubfinder/

Extract it into:

```text
res/tool/subfinder/
```

Then copy:

```text
res/tool/general.clg -> res/tool/subfinder/settings/general.cfg
```

The training pipeline on Linux does not need VideoSubFinder.
