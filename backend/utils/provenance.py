# backend/utils/provenance.py
"""
Embeds an artifact's provenance in its own file, so a file that leaves the project can be traced back
to how it was made when it is imported again.

WAV files carry it in a custom RIFF chunk and PNG files in an iTXt chunk. Other software skips both,
and neither touches the samples or pixels that an artifact's UID is hashed from.
"""
import json
import os
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Optional, Union

FORMAT_VERSION = 1
WAV_CHUNK_ID = b"stma"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_KEYWORD = b"stemma"

# Context key an imported copy uses to point back at the artifact it was exported from
ORIGIN_KEY = "origin_id"

# Job bookkeeping that means nothing once the job is over
_EXCLUDED_CONTEXT_KEYS = {"job_id", "execution_mode"}


def build_payload(element) -> dict:
    """The provenance record for an artifact: its identity and the context its edges are rebuilt from."""
    context = {k: v for k, v in (getattr(element, "context", None) or {}).items() if k not in _EXCLUDED_CONTEXT_KEYS}
    return {
        "stemma": FORMAT_VERSION,
        "id": element.id,
        "type": element.type,
        "name": element.name,
        "created": getattr(element, "created", None),
        "context": context,
    }


def embed_provenance(element, path: Union[str, Path, None] = None) -> bool:
    """
    Writes the element's provenance into its file, or into `path` if given, replacing any earlier record.
    Only call this on files the project owns (the readable copies and exports), never on a source file.
    Returns False if the file format is not supported or the file could not be written.
    """
    if path is None:
        asset = getattr(element, "file", None)
        path = getattr(asset, "path", None)
    if not path:
        return False
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".wav", ".png"):
        return False

    blob = json.dumps(build_payload(element), default=str).encode("utf-8")
    try:
        data = path.read_bytes()
        tagged = _tag_wav(data, blob) if suffix == ".wav" else _tag_png(data, blob)
        _replace_file(path, tagged)
        return True
    except Exception as e:
        print(f"Could not embed provenance in {path}: {e}")
        return False


def read_provenance(path: Union[str, Path]) -> Optional[dict]:
    """Returns the provenance record embedded in a WAV or PNG file, or None if it has none."""
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".wav":
            blob = _read_wav(path.read_bytes())
        elif suffix == ".png":
            blob = _read_png(path.read_bytes())
        else:
            return None
        if blob is None:
            return None
        payload = json.loads(blob.decode("utf-8"))
    except Exception as e:
        print(f"Ignoring unreadable provenance in {path}: {e}")
        return None
    if not isinstance(payload, dict) or payload.get("stemma") != FORMAT_VERSION:
        return None
    return payload


def restore_context(payload: dict, element_id: str) -> dict:
    """
    The context for an imported copy: the original's context, plus a pointer to the original itself.
    A re-encoded copy hashes to a new UID, so the pointer is what links the two.
    """
    context = dict(payload.get("context") or {})
    origin_id = payload.get("id")
    if isinstance(origin_id, str) and origin_id != element_id:
        context[ORIGIN_KEY] = origin_id
    return context


def _replace_file(path: Path, data: bytes) -> None:
    """Writes beside the file first, so a failed write never leaves a truncated asset."""
    fd, temp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(temp_path, path)
    except BaseException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


# --------------------
#  WAV (RIFF chunks)
# --------------------

def _riff_chunks(data: bytes):
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    i = 12
    while i + 8 <= len(data):
        chunk_id = data[i:i + 4]
        size = struct.unpack("<I", data[i + 4:i + 8])[0]
        yield chunk_id, data[i + 8:i + 8 + size]
        # Chunks are word-aligned
        i += 8 + size + (size & 1)


def _riff_chunk(chunk_id: bytes, body: bytes) -> bytes:
    return chunk_id + struct.pack("<I", len(body)) + body + (b"\0" if len(body) & 1 else b"")


def _tag_wav(data: bytes, blob: bytes) -> bytes:
    chunks = [_riff_chunk(cid, body) for cid, body in _riff_chunks(data) if cid != WAV_CHUNK_ID]
    form = b"WAVE" + b"".join(chunks) + _riff_chunk(WAV_CHUNK_ID, blob)
    return b"RIFF" + struct.pack("<I", len(form)) + form


def _read_wav(data: bytes) -> Optional[bytes]:
    for chunk_id, body in _riff_chunks(data):
        if chunk_id == WAV_CHUNK_ID:
            return body
    return None


# --------------------
#  PNG (iTXt chunk)
# --------------------

def _png_chunks(data: bytes):
    if data[:8] != PNG_SIGNATURE:
        raise ValueError("not a PNG file")
    i = 8
    while i + 8 <= len(data):
        size = struct.unpack(">I", data[i:i + 4])[0]
        chunk_type = data[i + 4:i + 8]
        yield chunk_type, data[i + 8:i + 8 + size]
        i += 12 + size


def _png_chunk(chunk_type: bytes, body: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + chunk_type + body + struct.pack(">I", crc)


def _is_provenance_itxt(chunk_type: bytes, body: bytes) -> bool:
    return chunk_type == b"iTXt" and body.startswith(PNG_KEYWORD + b"\0")


def _tag_png(data: bytes, blob: bytes) -> bytes:
    # keyword, compression flag and method, then empty language tag and translated keyword
    itxt = PNG_KEYWORD + b"\0" + b"\1\0" + b"\0" + b"\0" + zlib.compress(blob)
    out = [PNG_SIGNATURE]
    for chunk_type, body in _png_chunks(data):
        if _is_provenance_itxt(chunk_type, body):
            continue
        if chunk_type == b"IEND":
            out.append(_png_chunk(b"iTXt", itxt))
        out.append(_png_chunk(chunk_type, body))
    return b"".join(out)


def _read_png(data: bytes) -> Optional[bytes]:
    for chunk_type, body in _png_chunks(data):
        if not _is_provenance_itxt(chunk_type, body):
            continue
        rest = body[len(PNG_KEYWORD) + 1:]
        compressed = rest[0] == 1
        # Skip past the language tag and translated keyword
        _, _, rest = rest[2:].split(b"\0", 2)
        return zlib.decompress(rest) if compressed else rest
    return None
