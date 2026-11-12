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
from typing import BinaryIO, Collection, Sequence

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


def _read_header(f: BinaryIO, path: str) -> tuple[dict[str, dict], int]:
    """Returns the tensor entries of an open safetensors file and the offset its data starts at."""
    length_bytes = f.read(8)
    if len(length_bytes) != 8:
        raise ValueError(f"{path} is too short to be a safetensors file")
    header_length = struct.unpack("<Q", length_bytes)[0]
    if header_length > _MAX_HEADER_BYTES:
        raise ValueError(f"{path} has a {header_length}-byte header; it isn't a safetensors file")
    header = json.loads(f.read(header_length))
    header.pop("__metadata__", None)
    return header, 8 + header_length


def safetensors_names(path: str | os.PathLike) -> list[str]:
    """The names of the tensors in a safetensors file, read from its header alone."""
    path = os.fspath(path)
    with open(path, "rb") as f:
        return list(_read_header(f, path)[0])


def read_safetensors(path: str | os.PathLike, names: Collection[str] | None = None) -> dict[str, torch.Tensor]:
    """
    Loads the tensors in a safetensors file onto the CPU, reading the file sequentially once. With
    names, loads only those tensors (skipping the rest of the file) and raises if any is missing.
    """
    path = os.fspath(path)
    with open(path, "rb", buffering=0) as f:
        header, data_start = _read_header(f, path)
        if names is not None:
            missing = set(names) - set(header)
            if missing:
                raise KeyError(f"{path} has no tensors named {sorted(missing)}")
            header = {name: header[name] for name in names}

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


def materialize_from_safetensors(
    module: torch.nn.Module, files: Sequence[str | os.PathLike], key_prefixes: Sequence[str] = ("",)
) -> int:
    """
    Replaces the data of module's parameters and persistent buffers with tensors read sequentially
    from files, in place, so tied weights stay tied. A library that loaded the module from the same
    files may have left its weights memory-mapped; this swaps them for weights held in memory.

    Each weight is looked up in the files as one of key_prefixes followed by its name in module (for
    example "model." for a Hugging Face model saved with its base_model_prefix). Raises if any weight
    is missing from the files or has a different shape there. Returns the number of bytes read.
    """
    # Every name each distinct tensor goes by, since a tied weight may be saved under any of them
    persistent = set(module.state_dict().keys())
    aliases: dict[int, tuple[torch.Tensor, list[str]]] = {}
    named = list(module.named_parameters(remove_duplicate=False)) + [
        (name, buffer) for name, buffer in module.named_buffers(remove_duplicate=False) if name in persistent
    ]
    for name, tensor in named:
        aliases.setdefault(id(tensor), (tensor, []))[1].append(name)

    file_of: dict[str, str] = {}
    for file in files:
        for key in safetensors_names(file):
            file_of.setdefault(key, os.fspath(file))

    # file -> key in that file -> the module tensors it fills
    plan: dict[str, dict[str, list[tuple[str, torch.Tensor]]]] = {}
    missing = []
    for tensor, names in aliases.values():
        name = names[0]
        key = next(
            (prefix + alias for alias in names for prefix in key_prefixes if prefix + alias in file_of), None
        )
        if key is None:
            missing.append(name)
            continue
        plan.setdefault(file_of[key], {}).setdefault(key, []).append((name, tensor))
    if missing:
        raise KeyError(
            f"{len(missing)} weight(s) not found in {[os.fspath(f) for f in files]} under prefixes "
            f"{list(key_prefixes)}, e.g. {missing[:3]}"
        )

    total = 0
    for file, wanted in plan.items():
        loaded = read_safetensors(file, names=wanted.keys())
        for key, fills in wanted.items():
            source = loaded.pop(key)
            for name, tensor in fills:
                if source.shape != tensor.shape:
                    raise ValueError(
                        f"'{key}' in {file} has shape {tuple(source.shape)}, but '{name}' is {tuple(tensor.shape)}"
                    )
                tensor.data = source.to(tensor.dtype)
                total += tensor.numel() * tensor.element_size()
    return total
