"""
Receives uploads into the engine service's content-addressed cache (<data>/cache/xx/<uid>).

A file asset arrives in chunks at <uid>.part and is renamed into place after its last chunk. A
directory asset is stored as the directory itself, never as an archive, so its UID and recorded size
describe what is on disk. Each of its files arrives the same way under <uid>.part/<relpath>, and
finalize_directory() then checks the staged directory against its UID and renames it into place.

Nothing is renamed into place until it is complete, so a reader never sees half an asset. Staging
left by a failed upload is removed by the cache sweep (see cache_policy.py).
"""
import os
from pathlib import Path, PurePosixPath

from utils.filesystem import get_path_size, remove_path
from utils.uid import XXH3_64, path_from_uid


class UploadError(Exception):
    """An upload was rejected; the client should send it again from the start."""


def destination_path(data_root: Path, uid: str) -> Path:
    return Path(data_root) / path_from_uid(uid)


def staging_path(data_root: Path, uid: str) -> Path:
    return Path(data_root) / f"{path_from_uid(uid)}.part"


def _member_path(root: Path, relpath: str) -> Path:
    """Where a directory member lands under root. Rejects paths that would escape it."""
    member = PurePosixPath(relpath)
    if not relpath or member.is_absolute() or ".." in member.parts or ":" in relpath or "\\" in relpath:
        raise UploadError(f"Invalid path inside directory asset: '{relpath}'")
    return root.joinpath(*member.parts)


def write_chunk(
    data_root: Path,
    uid: str,
    chunk: bytes,
    chunk_index: int,
    total_chunks: int,
    total_size: int,
    relpath: str | None = None,
) -> bool:
    """
    Writes one chunk of a file asset, or of one file inside a directory asset when relpath is given
    (total_size is then that file's size). Returns True once the file is complete. A complete file
    asset is moved into place; a directory asset waits for finalize_directory(). Callers serialize
    calls for the same UID.
    """
    staging = staging_path(data_root, uid)
    if relpath is None:
        if staging.is_dir():
            remove_path(staging)
        target = staging
    else:
        if staging.is_file():
            staging.unlink()
        target = _member_path(staging, relpath)
    target.parent.mkdir(parents=True, exist_ok=True)

    with open(target, "wb" if chunk_index == 0 else "ab") as f:
        f.write(chunk)
    if chunk_index < total_chunks - 1:
        return False

    written = target.stat().st_size
    if total_size > 0 and written != total_size:
        target.unlink(missing_ok=True)
        raise UploadError(f"Size mismatch for '{relpath or uid}': expected {total_size} bytes, got {written}")

    if relpath is None:
        destination = destination_path(data_root, uid)
        remove_path(destination)
        os.replace(staging, destination)
    return True


def finalize_directory(data_root: Path, uid: str, total_size: int) -> None:
    """
    Checks a staged directory asset against its size and UID and moves it into place. On a mismatch
    the staging is discarded, so a retry starts clean. Callers serialize calls for the same UID.
    """
    staging = staging_path(data_root, uid)
    destination = destination_path(data_root, uid)
    if not staging.is_dir():
        if total_size == 0:
            staging.mkdir(parents=True, exist_ok=True)
        else:
            raise UploadError(f"No staged files for directory asset '{uid}'")

    try:
        staged_size = get_path_size(staging)
        if total_size > 0 and staged_size != total_size:
            raise UploadError(f"Size mismatch for '{uid}': expected {total_size} bytes, got {staged_size}")
        computed = XXH3_64().from_directory(staging)
        if computed != uid:
            raise UploadError(f"Directory contents hash to '{computed}', not '{uid}'")
    except UploadError:
        remove_path(staging)
        raise

    remove_path(destination)
    os.replace(staging, destination)
