# backend/utils/safetensors_io.py
"""
Reads a safetensors file into memory with large sequential reads.

safetensors' own load_file memory-maps the file, so its tensors are views onto it that are only read
when touched. On a slow or remote filesystem (such as the 9p share Docker Desktop uses for Windows
folders) touching them pulls the file in through page faults of a few KB each, which can be many
times slower than reading it straight through. Here each tensor gets its own memory, read in one
pass in file order.
"""
import json
import os
import struct
from typing import BinaryIO

import torch

# safetensors dtype names, as written in the header
_DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
for _name, _attr in (("F8_E4M3", "float8_e4m3fn"), ("F8_E5M2", "float8_e5m2"), ("U16", "uint16"),
                     ("U32", "uint32"), ("U64", "uint64")):
    if hasattr(torch, _attr):
        _DTYPES[_name] = getattr(torch, _attr)

# The header is JSON; anything near this size means the file isn't safetensors
_MAX_HEADER_BYTES = 100_000_000


def _read_exact(f: BinaryIO, buffer: memoryview, path: str) -> None:
    filled = 0
    while filled < len(buffer):
        count = f.readinto(buffer[filled:])
        if not count:
            raise ValueError(f"Unexpected end of safetensors file {path}")
        filled += count


def read_safetensors(path: str | os.PathLike) -> dict[str, torch.Tensor]:
    """Loads every tensor in a safetensors file onto the CPU, reading the file sequentially once."""
    path = os.fspath(path)
    with open(path, "rb", buffering=0) as f:
        length_bytes = f.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"{path} is too short to be a safetensors file")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if header_length > _MAX_HEADER_BYTES:
            raise ValueError(f"{path} has a {header_length}-byte header; it isn't a safetensors file")
        header = json.loads(f.read(header_length))
        header.pop("__metadata__", None)
        data_start = 8 + header_length

        # File order, so the reads stay sequential
        entries = sorted(header.items(), key=lambda item: item[1]["data_offsets"][0])
        tensors: dict[str, torch.Tensor] = {}
        for name, info in entries:
            dtype = _DTYPES.get(info["dtype"])
            if dtype is None:
                raise ValueError(f"Tensor '{name}' in {path} has unsupported dtype {info['dtype']}")
            begin, end = info["data_offsets"]
            raw = torch.empty(end - begin, dtype=torch.uint8)
            if end > begin:
                f.seek(data_start + begin)
                _read_exact(f, memoryview(raw.numpy()).cast("B"), path)
            tensors[name] = raw.view(dtype).reshape(info["shape"])
        return tensors
