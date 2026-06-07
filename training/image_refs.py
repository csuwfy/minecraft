"""Image reference loading helpers for manifests."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pyarrow.parquet as pq
from PIL import Image

TESS_STAGE1_SCHEME = "tess_stage1://"


class TessStage1ImageStore:
    """Lazy image loader for TESS Stage1 parquet-backed frame references."""

    def __init__(self, index_path: str | Path, max_cache_groups: int = 2) -> None:
        self.index_path = Path(index_path)
        index = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.parquet_files = [Path(path) for path in index["parquet_files"]]
        self.max_cache_groups = max(1, int(os.environ.get("TESS_STAGE1_IMAGE_CACHE_GROUPS", max_cache_groups)))
        self._parquet_files: Dict[int, pq.ParquetFile] = {}
        self._row_groups: OrderedDict[tuple[int, int], Any] = OrderedDict()

    def load(self, ref: str) -> Image.Image:
        if not ref.startswith(TESS_STAGE1_SCHEME):
            raise ValueError(f"Unsupported TESS Stage1 ref: {ref}")
        payload = ref[len(TESS_STAGE1_SCHEME) :]
        file_index_text, row_text = payload.split("/", 1)
        file_index = int(file_index_text)
        row_index = int(row_text)
        parquet_file = self._parquet_file(file_index)
        row_group_index, row_offset = self._locate_row_group(parquet_file, row_index)
        table = self._row_group(file_index, row_group_index, parquet_file)
        value = table.column("image")[row_offset].as_py()
        return image_from_value(value)

    def _parquet_file(self, file_index: int) -> pq.ParquetFile:
        parquet_file = self._parquet_files.get(file_index)
        if parquet_file is None:
            parquet_file = pq.ParquetFile(self.parquet_files[file_index])
            self._parquet_files[file_index] = parquet_file
        return parquet_file

    @staticmethod
    def _locate_row_group(parquet_file: pq.ParquetFile, row_index: int) -> tuple[int, int]:
        remaining = row_index
        for group_index in range(parquet_file.num_row_groups):
            rows = parquet_file.metadata.row_group(group_index).num_rows
            if remaining < rows:
                return group_index, remaining
            remaining -= rows
        raise IndexError(f"Row index {row_index} out of range")

    def _row_group(self, file_index: int, row_group_index: int, parquet_file: pq.ParquetFile):
        key = (file_index, row_group_index)
        table = self._row_groups.get(key)
        if table is None:
            table = parquet_file.read_row_group(row_group_index, columns=["image"])
            self._row_groups[key] = table
            while len(self._row_groups) > self.max_cache_groups:
                self._row_groups.popitem(last=False)
        else:
            self._row_groups.move_to_end(key)
        return table


def image_from_value(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        return Image.open(BytesIO(value)).convert("RGB")
    if isinstance(value, bytearray):
        return Image.open(BytesIO(bytes(value))).convert("RGB")
    if isinstance(value, Mapping):
        raw = value.get("bytes")
        if isinstance(raw, bytes):
            return Image.open(BytesIO(raw)).convert("RGB")
        path = value.get("path")
        if isinstance(path, str) and path:
            return Image.open(path).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def is_tess_stage1_ref(value: str) -> bool:
    return value.startswith(TESS_STAGE1_SCHEME)


def load_image(value: Any, tess_stage1_store: Optional[TessStage1ImageStore] = None) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, str) and is_tess_stage1_ref(value):
        if tess_stage1_store is None:
            raise ValueError("TESS Stage1 image ref requires a TessStage1ImageStore.")
        return tess_stage1_store.load(value)
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")
    return image_from_value(value)
