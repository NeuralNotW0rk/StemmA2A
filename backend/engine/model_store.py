"""
Durable storage for shared models, kept apart from the evictable asset cache.

    <data>/cache/xx/<uid>              anonymous, size-capped, evictable (see cache_policy.py)
    <data>/models/xx/<uid>             model files, never evicted
    <data>/models/shared_models.json   the registry: named references to model files

A registry entry:

    {
      "name": "Stable Audio 3 Medium",
      "adapter": "stable_audio_tools",
      "model_type": "stable_audio_tools",
      "checkpoint": {"uid": "f28c...xxh3_64", "size": 9222116660},
      "encoder": {"uid": "1751...xxh3_64", "size": 1221696011},   # optional
      "config": {...}                                              # inline, or
      "config_file": {"uid": "..."}                                # a file reference
    }

A file reference names a file in the store by "uid". It can give a "path" instead, for a file that
already exists on this machine and is used where it is; a "uid" next to a path saves recomputing it.
That is the only place paths appear. Files enter the store by promotion, which moves them out of the
cache, and leave it by demotion, which moves them back so they become ordinary evictable entries.
"""
import json
import os
import shutil
import uuid
import zipfile
from pathlib import Path

from utils.uid import path_from_uid

MODELS_DIR = "models"
REGISTRY_FILE = "shared_models.json"
# Entry keys that hold file references, and the register_model() keyword prefix for each
FILE_REFS = {"checkpoint": "checkpoint", "encoder": "encoder", "config_file": "config"}


class MissingModelFiles(Exception):
    """A promoted entry names UIDs that are in neither the store nor the cache."""

    def __init__(self, uids: list[str]):
        super().__init__(f"Missing model files: {uids}")
        self.uids = uids


def data_root_from_env() -> Path:
    """The data root the engine stores assets under, from the same variables the servers use."""
    if os.environ.get("RUNNING_IN_CONTAINER") == "true":
        return Path(os.environ.get("CONTAINER_DATA_PATH") or "/data").expanduser()
    return Path(os.environ.get("LOCAL_DATA_PATH") or "~/.stemma2a_data").expanduser()


def _entry_key(entry: dict) -> tuple:
    """Entries naming the same checkpoint and encoder are the same shared model."""
    def ref_key(ref):
        return (ref.get("uid") or ref.get("path")) if isinstance(ref, dict) else None
    return ref_key(entry.get("checkpoint")), ref_key(entry.get("encoder"))


def _sizes_match(path: Path, size: int | None) -> bool:
    # Directories are uploaded zipped, so a zip's size won't match the directory's recorded size
    return not size or path.stat().st_size == size or zipfile.is_zipfile(path)


class ModelStore:
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.models_dir = self.data_root / MODELS_DIR
        self.registry_path = self.models_dir / REGISTRY_FILE

    def model_path(self, uid: str) -> Path:
        # path_from_uid gives cache/xx/<uid>; the store uses the same sharding under models/
        return self.models_dir.joinpath(*path_from_uid(uid).parts[1:])

    def cache_path(self, uid: str) -> Path:
        return self.data_root / path_from_uid(uid)

    def resolve(self, uid: str) -> Path | None:
        """Where a UID's file is: the store first, then the cache."""
        for path in (self.model_path(uid), self.cache_path(uid)):
            if path.exists():
                return path
        return None

    # Registry

    def load_registry(self) -> list[dict]:
        if not self.registry_path.exists():
            return []
        with open(self.registry_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            raise ValueError(f"{self.registry_path} must contain a list of entries")
        return entries

    def save_registry(self, entries: list[dict]) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self.registry_path.with_name(f"{REGISTRY_FILE}.{uuid.uuid4().hex[:8]}.tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, self.registry_path)

    def resolve_ref(self, ref: dict) -> Path:
        """The file a reference points at. Raises if it isn't there."""
        if ref.get("path"):
            path = Path(ref["path"])
            if path.exists():
                return path
            if not ref.get("uid"):
                raise FileNotFoundError(f"Model file not found: {path}")
        if ref.get("uid"):
            path = self.model_path(ref["uid"])
            if path.exists():
                return path
            raise FileNotFoundError(f"Model file '{ref['uid']}' is not in the model store")
        raise ValueError(f"File reference needs a 'uid' or a 'path': {ref}")

    def register_kwargs(self, entry: dict) -> dict:
        """Keyword arguments for an adapter's register_model(), with every file reference resolved."""
        if not entry.get("checkpoint"):
            raise ValueError(f"Shared model '{entry.get('name')}' has no checkpoint")
        kwargs = {
            "name": entry.get("name"),
            "model_type": entry.get("model_type"),
            "config": entry.get("config"),
        }
        for key, prefix in FILE_REFS.items():
            ref = entry.get(key)
            if not ref:
                continue
            kwargs[f"{prefix}_path"] = str(self.resolve_ref(ref))
            if prefix != "config":
                kwargs[f"{prefix}_uid"] = ref.get("uid")
                kwargs[f"{prefix}_size"] = ref.get("size")
        return kwargs

    # Promotion and demotion

    def promote(self, entry: dict) -> dict:
        """
        Adds or replaces a registry entry, moving the files it names from the cache into the store.
        A reference keeps its path only if that path exists here; otherwise it must name a UID that
        is in the store or the cache. Raises MissingModelFiles, changing nothing, if any are absent.
        """
        normalized = {k: v for k, v in entry.items() if k not in FILE_REFS}
        to_move: list[tuple[Path, Path]] = []
        missing = []
        for key in FILE_REFS:
            ref = entry.get(key)
            if not ref:
                continue
            uid, size, path = ref.get("uid"), ref.get("size"), ref.get("path")
            if path and Path(path).exists():
                normalized[key] = {k: v for k, v in (("path", path), ("uid", uid), ("size", size)) if v}
                continue
            if not uid:
                raise ValueError(f"'{key}' needs a 'uid', or a 'path' that exists on the engine host")
            normalized[key] = {k: v for k, v in (("uid", uid), ("size", size)) if v}
            stored, cached = self.model_path(uid), self.cache_path(uid)
            if stored.exists():
                continue
            if not cached.exists():
                missing.append(uid)
            elif not _sizes_match(cached, size):
                raise ValueError(
                    f"Cached file for '{uid}' is {cached.stat().st_size} bytes, expected {size}"
                )
            else:
                to_move.append((cached, stored))
        if missing:
            raise MissingModelFiles(missing)
        if not normalized.get("checkpoint"):
            raise ValueError("A shared model needs a checkpoint")

        for source, destination in to_move:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)

        key = _entry_key(normalized)
        entries = [e for e in self.load_registry() if _entry_key(e) != key]
        entries.append(normalized)
        self.save_registry(entries)
        return normalized

    def demote(self, entry: dict) -> None:
        """
        Removes the entry from the registry and moves its store files back into the cache, unless
        another entry still references them. Files referenced by path are left alone.
        """
        key = _entry_key(entry)
        entries = self.load_registry()
        remaining = [e for e in entries if _entry_key(e) != key]
        if len(remaining) == len(entries):
            raise ValueError(f"Shared model '{entry.get('name')}' is not in the registry")
        self.save_registry(remaining)

        still_referenced = {
            ref.get("uid") for e in remaining for k in FILE_REFS if isinstance(ref := e.get(k), dict)
        }
        for k in FILE_REFS:
            ref = entry.get(k)
            uid = ref.get("uid") if isinstance(ref, dict) else None
            if not uid or uid in still_referenced:
                continue
            stored = self.model_path(uid)
            if not stored.exists():
                continue
            cached = self.cache_path(uid)
            cached.parent.mkdir(parents=True, exist_ok=True)
            if cached.exists():
                stored.unlink()
            else:
                os.replace(stored, cached)
                # Back in the cache as if just used, so it ages out under the normal policy
                os.utime(cached, None)


def migrate_legacy_registry(data_root: Path, legacy_path: Path) -> bool:
    """
    Converts a legacy shared_models.json (flat checkpoint_uid/checkpoint_path keys, or nodes copied
    from graph.json) into the registry, moving any files it found in the cache into the store. Does
    nothing if the registry already exists. Returns whether it migrated.
    """
    store = ModelStore(data_root)
    if store.registry_path.exists() or not legacy_path.exists():
        return False

    with open(legacy_path, "r", encoding="utf-8") as f:
        legacy = json.load(f)

    def migrate_ref(uid, path, size) -> dict | None:
        if not uid and not path:
            return None
        if uid:
            stored, cached = store.model_path(uid), store.cache_path(uid)
            if cached.exists() and not stored.exists():
                stored.parent.mkdir(parents=True, exist_ok=True)
                os.replace(cached, stored)
            if stored.exists():
                return {k: v for k, v in (("uid", uid), ("size", size)) if v}
        # Kept even if the path is missing here, so the loader reports the entry rather than losing it
        return {k: v for k, v in (("path", path), ("uid", uid), ("size", size)) if v}

    entries = []
    for entry in legacy:
        if isinstance(entry.get("data"), dict):
            entry = entry["data"]
        migrated = {k: entry[k] for k in ("name", "adapter", "model_type") if entry.get(k)}
        for key in ("checkpoint", "encoder"):
            nested = entry.get(key) if isinstance(entry.get(key), dict) else {}
            ref = migrate_ref(
                nested.get("uid") or entry.get(f"{key}_uid"),
                nested.get("path") or entry.get(f"{key}_path"),
                nested.get("size") or entry.get(f"{key}_size"),
            )
            if ref:
                migrated[key] = ref
        if isinstance(entry.get("config"), dict):
            migrated["config"] = entry["config"]
        config_ref = migrate_ref(entry.get("config_uid"), entry.get("config_path"), None)
        if config_ref:
            migrated["config_file"] = config_ref
        entries.append(migrated)

    store.save_registry(entries)
    shutil.move(str(legacy_path), str(legacy_path.with_name(f"{legacy_path.name}.migrated")))
    return True
