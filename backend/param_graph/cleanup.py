"""
Finds project files that nothing in the graph references, and moves them to a restorable trash
folder inside the project before anything is deleted for good.

A file counts as referenced when any string in the graph could point at it. Strings come from the
in-memory graph, graph.json and graph.json.bak, since load() falls back to the backup. Matching is
deliberately loose so that a doubtful file is kept, not removed:

- Paths are compared from the managed directory onward ("generate/foo.wav"), so stored absolute
  paths still match after the project folder is moved or renamed.
- A file whose name, with any of its suffixes stripped, equals some string in the graph is kept.
  This covers CAS copies (cache/xx/<uid>), cached external audio (cache/<node_id>.wav) and genomes
  named by their UID.
"""
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .const import DICT_FILE

# Project subdirectories that hold generated or cached files; nothing else in the project is touched
MANAGED_DIRS = ("generate", "process", "cache")
TRASH_DIR = ".trash"
MANIFEST_FILE = "manifest.json"
_BATCH_ID = re.compile(r"^[0-9]{8}-[0-9]{6}(-[0-9]+)?$")


@dataclass
class References:
    # Lowercased POSIX paths from a managed directory onward, e.g. "generate/foo.wav"
    relative: set[str] = field(default_factory=set)
    # Every string value in the graph (ids, uids, names, ...)
    tokens: set[str] = field(default_factory=set)

    def add_string(self, value: str) -> None:
        self.tokens.add(value)
        if "/" in value or "\\" in value:
            self.relative.update(_managed_tails(value))


def _managed_tails(path_str: str) -> list[str]:
    """Every suffix of the path that starts at a managed directory name."""
    parts = [p for p in path_str.replace("\\", "/").split("/") if p]
    return [
        "/".join(parts[i:]).lower()
        for i in range(len(parts) - 1)
        if parts[i].lower() in MANAGED_DIRS
    ]


def _iter_strings(value) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                yield key
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_strings(item)


def collect_references(param_graph) -> References:
    """Gathers references from the in-memory graph and from graph.json and its backup on disk."""
    refs = References()
    G = param_graph.G
    for node, data in G.nodes(data=True):
        refs.add_string(str(node))
        for value in _iter_strings(data):
            refs.add_string(value)
    for _, _, data in G.edges(data=True):
        for value in _iter_strings(data):
            refs.add_string(value)

    for file_name in (DICT_FILE, f"{DICT_FILE}.bak"):
        graph_file = Path(param_graph.root) / file_name
        if not graph_file.exists() or graph_file.stat().st_size == 0:
            continue
        try:
            with open(graph_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            # Without every reference source we can't tell what is safe to remove
            raise ValueError(f"Cannot read '{graph_file}' to check file references: {e}") from e
        for value in _iter_strings(data):
            refs.add_string(value)
    return refs


def _name_candidates(name: str) -> list[str]:
    """The file name with zero, one, two... suffixes stripped: a.xxh3_64.wav -> a.xxh3_64.wav, a.xxh3_64, a."""
    candidates = [name]
    while "." in name.lstrip("."):
        name = name.rsplit(".", 1)[0]
        candidates.append(name)
    return candidates


def is_referenced(relative_path: str, refs: References) -> bool:
    if relative_path.lower() in refs.relative:
        return True
    return any(c in refs.tokens for c in _name_candidates(relative_path.rsplit("/", 1)[-1]))


@dataclass
class OrphanFile:
    path: str  # POSIX path relative to the project root
    size: int


def find_unreferenced(root: Path, refs: References) -> list[OrphanFile]:
    """Files under the managed directories that nothing references, sorted by path."""
    root = Path(root)
    orphans = []
    for managed in MANAGED_DIRS:
        managed_dir = root / managed
        if not managed_dir.is_dir():
            continue
        for dir_path, _, file_names in os.walk(managed_dir, followlinks=False):
            for file_name in file_names:
                file_path = Path(dir_path) / file_name
                if file_path.is_symlink():
                    continue
                relative = file_path.relative_to(root).as_posix()
                if not is_referenced(relative, refs):
                    orphans.append(OrphanFile(path=relative, size=file_path.stat().st_size))
    orphans.sort(key=lambda o: o.path)
    return orphans


def _prune_empty_dirs(start: Path, stop: Path) -> None:
    """Removes empty directories from start up to, but not including, stop."""
    current = start
    while current != stop and stop in current.parents:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def quarantine(root: Path, orphans: list[OrphanFile]) -> dict:
    """
    Moves the files into .trash/<batch_id>/, keeping their relative paths, and writes a manifest so
    they can be restored. The manifest lists exactly the files moved, even if a move fails part way.
    """
    root = Path(root)
    trash_root = root / TRASH_DIR
    batch_id = time.strftime("%Y%m%d-%H%M%S")
    counter = 1
    while (trash_root / batch_id).exists():
        batch_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{counter}"
        counter += 1
    batch_dir = trash_root / batch_id
    batch_dir.mkdir(parents=True)

    moved = []
    try:
        for orphan in orphans:
            source = root / orphan.path
            destination = batch_dir / orphan.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            moved.append({"path": orphan.path, "size": orphan.size})
            _prune_empty_dirs(source.parent, root / Path(orphan.path).parts[0])
    finally:
        manifest = {"created": int(time.time()), "files": moved}
        with open(batch_dir / MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)

    return {"id": batch_id, "file_count": len(moved), "bytes": sum(m["size"] for m in moved)}


def _batch_dir(root: Path, batch_id: str) -> Path:
    if not _BATCH_ID.match(batch_id or ""):
        raise ValueError(f"Invalid trash batch id '{batch_id}'")
    batch_dir = Path(root) / TRASH_DIR / batch_id
    if not batch_dir.is_dir():
        raise FileNotFoundError(f"Trash batch '{batch_id}' not found")
    return batch_dir


def _read_manifest(batch_dir: Path) -> dict:
    manifest_path = batch_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise ValueError(f"Trash batch '{batch_dir.name}' has no {MANIFEST_FILE}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_trash(root: Path) -> list[dict]:
    """Trash batches, newest first. Counts and sizes come from what is on disk now."""
    trash_root = Path(root) / TRASH_DIR
    if not trash_root.is_dir():
        return []
    batches = []
    for batch_dir in trash_root.iterdir():
        if not batch_dir.is_dir() or not _BATCH_ID.match(batch_dir.name):
            continue
        manifest = _read_manifest(batch_dir)
        files = [p for p in batch_dir.rglob("*") if p.is_file() and p.name != MANIFEST_FILE]
        batches.append({
            "id": batch_dir.name,
            "created": manifest.get("created"),
            "file_count": len(files),
            "bytes": sum(p.stat().st_size for p in files),
        })
    batches.sort(key=lambda b: b["id"], reverse=True)
    return batches


def restore(root: Path, batch_id: str) -> dict:
    """
    Moves a batch's files back to where they were. A file whose original location is occupied again
    is left in the trash and reported as a conflict, never overwritten.
    """
    root = Path(root)
    batch_dir = _batch_dir(root, batch_id)
    manifest = _read_manifest(batch_dir)

    restored, conflicts, remaining = [], [], []
    for entry in manifest["files"]:
        source = batch_dir / entry["path"]
        destination = root / entry["path"]
        if not source.exists():
            continue
        if destination.exists():
            conflicts.append(entry["path"])
            remaining.append(entry)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        restored.append(entry["path"])
        _prune_empty_dirs(source.parent, batch_dir)

    if remaining:
        manifest["files"] = remaining
        with open(batch_dir / MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
    else:
        shutil.rmtree(batch_dir)
        _prune_empty_dirs(root / TRASH_DIR, root)
    return {"restored": len(restored), "conflicts": conflicts}


def purge(root: Path, batch_id: str) -> dict:
    """Permanently deletes a trash batch."""
    root = Path(root)
    batch_dir = _batch_dir(root, batch_id)
    files = [p for p in batch_dir.rglob("*") if p.is_file() and p.name != MANIFEST_FILE]
    freed = sum(p.stat().st_size for p in files)
    shutil.rmtree(batch_dir)
    _prune_empty_dirs(root / TRASH_DIR, root)
    return {"file_count": len(files), "bytes": freed}
