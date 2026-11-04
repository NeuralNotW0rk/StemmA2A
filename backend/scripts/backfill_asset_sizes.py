"""
Records the missing byte sizes of model checkpoints and encoders in projects.

Adapters only trust a model's UID without hashing when the file's size matches the size recorded
with it (see StableAudioAdapter.load_model). Models imported before sizes were recorded have none,
so the engine re-hashes the whole checkpoint every time it loads one. This script hashes each such
file once and, only if that reproduces the stored UID, records its current size. A file that no
longer hashes to its UID is reported and left alone, so the size never vouches for different content.

Close the project in the app first; it would save its in-memory graph over the rewritten one. The
original graph.json is kept as graph.json.pre-size-backfill.

    python scripts/backfill_asset_sizes.py --dry-run "<workspace>/projects"
    python scripts/backfill_asset_sizes.py "<workspace>/projects/demo"
"""
import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Callable, Iterator

# Ensure we can import from the backend directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.filesystem import get_path_size
from utils.uid import XXH3_64

BACKUP_NAME = "graph.json.pre-size-backfill"

# Computes the UID an asset at a path was registered under
Hasher = Callable[[Path], str]


def default_hashers(uid_gen: XXH3_64) -> dict[tuple[str, str], Hasher]:
    """
    (adapter, asset field) -> the hash that adapter's register_model gives that asset. The adapters
    are imported only when hashed, since they pull in torch.
    """
    def stable_audio_checkpoint(path: Path) -> str:
        from engine.model_adapters.stable_audio_adapter import _compute_stable_audio_uid
        return _compute_stable_audio_uid(str(path), uid_gen)

    def stylegan_checkpoint(path: Path) -> str:
        from engine.model_adapters.stylegan_adapter import _compute_stylegan_uid
        return _compute_stylegan_uid(str(path), uid_gen)

    def stable_audio_encoder(path: Path) -> str:
        if path.is_dir():
            return uid_gen.from_directory(path)
        # A single-file encoder is registered under a hash of its path
        return uid_gen.from_string(str(path))

    return {
        ("stable_audio_tools", "checkpoint"): stable_audio_checkpoint,
        ("stable_audio_tools", "encoder"): stable_audio_encoder,
        ("stylegan2", "checkpoint"): stylegan_checkpoint,
    }


def unsized_assets(node: dict) -> Iterator[tuple[str, dict]]:
    """(field name, asset dict) for each of a model node's assets with a UID and path but no size."""
    if node.get("type") != "model":
        return
    for key in ("checkpoint", "encoder"):
        asset = node.get(key)
        if isinstance(asset, dict) and asset.get("uid") and asset.get("path") and not asset.get("size"):
            yield key, asset


def plan(
    nodes: list[dict],
    hashers: dict[tuple[str, str], Hasher],
    verified: dict[tuple[str, str], int | None],
) -> list[tuple[dict, int]]:
    """
    (asset dict, size) for every unsized model asset whose file still hashes to its UID. verified
    caches (path, uid) -> size, or None when the file didn't match, across projects sharing a model.
    """
    updates: list[tuple[dict, int]] = []
    for node in nodes:
        label = node.get("name") or node.get("id")
        for key, asset in unsized_assets(node):
            path = Path(asset["path"])
            if not path.exists():
                print(f"  Skipping {label} {key}: {path} doesn't exist here")
                continue
            hasher = hashers.get((node.get("adapter"), key))
            if hasher is None:
                print(f"  Skipping {label} {key}: no known hash for adapter '{node.get('adapter')}'")
                continue
            cache_key = (os.path.normcase(os.path.abspath(path)), asset["uid"])
            if cache_key not in verified:
                print(f"  Hashing {path}...")
                matches = hasher(path) == asset["uid"]
                verified[cache_key] = get_path_size(path) if matches else None
            size = verified[cache_key]
            if size is None:
                print(f"  Skipping {label} {key}: {path} no longer hashes to '{asset['uid']}'; it may have changed since import")
                continue
            updates.append((asset, size))
    return updates


def backfill(
    project: Path,
    dry_run: bool,
    verified: dict[tuple[str, str], int | None],
    hashers: dict[tuple[str, str], Hasher] | None = None,
) -> bool:
    graph_file = project / "graph.json"
    with open(graph_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = [n["data"] for n in data.get("graph", {}).get("elements", {}).get("nodes", []) if "data" in n]
    updates = plan(nodes, hashers if hashers is not None else default_hashers(XXH3_64()), verified)
    if not updates:
        return False
    for asset, size in updates:
        print(f"  {asset['uid']}: {size} bytes")
    if dry_run:
        return True

    for asset, size in updates:
        asset["size"] = size
    backup = project / BACKUP_NAME
    if not backup.exists():
        shutil.copy2(graph_file, backup)
    temp = project / f"graph.json.{uuid.uuid4().hex[:8]}.tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, graph_file)
    print(f"  Rewrote {graph_file} (original kept at {BACKUP_NAME})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="Project folders, or folders of projects")
    parser.add_argument("--dry-run", action="store_true", help="Hash and report what would change without writing anything")
    args = parser.parse_args()

    projects = []
    for path in args.paths:
        if (path / "graph.json").is_file():
            projects.append(path)
        elif path.is_dir():
            projects += sorted(p.parent for p in path.glob("*/graph.json"))
        else:
            sys.exit(f"Error: {path} is not a project or a folder of projects")

    # Models shared by several projects are hashed once
    verified: dict[tuple[str, str], int | None] = {}
    changed = 0
    for project in projects:
        print(f"{project.name}:")
        if backfill(project, args.dry_run, verified):
            changed += 1
        else:
            print("  Nothing to change.")
    verb = "would change" if args.dry_run else "changed"
    print(f"{changed} of {len(projects)} project(s) {verb}.")


if __name__ == "__main__":
    main()
