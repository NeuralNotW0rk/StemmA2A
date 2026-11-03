"""
Rewrites directory asset UIDs in projects to the current from_directory hash.

Before 2026-07-21, directories (such as Stable Audio encoders) were hashed by an older function that
read every file as raw bytes. from_directory normalizes JSON and text files first, so the same
directory now hashes differently, and the engine service rejects a directory whose contents don't
hash to its UID. This script finds directory assets whose UID came from the old hash, gives each the
current one, and renames the models whose IDs were built from it (their ID combines checkpoint and
encoder UIDs). Every reference in graph.json is rewritten, including edges.

A UID is only changed after checking that the old hash of the directory reproduces it, so a directory
that changed since it was imported is reported and left alone. Close the project in the app first;
it would save its in-memory graph over the rewritten one. The original graph.json is kept as
graph.json.pre-directory-uid-migration.

    python scripts/migrate_directory_uids.py --dry-run "<workspace>/projects"
    python scripts/migrate_directory_uids.py "<workspace>/projects/dagei"
"""
import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import xxhash

# Ensure we can import from the backend directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.migrations import readdress
from utils.uid import XXH3_64

BACKUP_NAME = "graph.json.pre-directory-uid-migration"


def legacy_directory_uid(directory: Path) -> str:
    """The pre-2026-07-21 directory hash (StableAudioAdapter._hash_directory), kept only to recognize its UIDs."""
    h = xxhash.xxh3_64()
    for file_path in sorted(directory.glob("**/*")):
        if file_path.is_file():
            h.update(str(file_path.relative_to(directory)).encode("utf-8"))
            with open(file_path, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    h.update(chunk)
    return f"{h.hexdigest()}.xxh3_64"


def directory_assets(node: dict):
    """(field name, asset dict) for each of a node's assets that names an existing directory."""
    for key, value in node.items():
        if isinstance(value, dict) and value.get("uid") and value.get("path") and Path(value["path"]).is_dir():
            yield key, value


def plan(nodes: dict[str, dict], uid_gen: XXH3_64, hashed: dict[str, tuple[str, str]]) -> dict[str, str]:
    """Old ID -> new ID for every legacy directory UID in the graph and every model ID built from one."""
    renamed: dict[str, str] = {}
    for node_id, node in nodes.items():
        for key, asset in directory_assets(node):
            old_uid = asset["uid"]
            if old_uid in renamed:
                continue
            directory = Path(asset["path"])
            cache_key = os.path.normcase(os.path.abspath(directory))
            if cache_key not in hashed:
                print(f"  Hashing {directory}...")
                hashed[cache_key] = (uid_gen.from_directory(directory), legacy_directory_uid(directory))
            current, legacy = hashed[cache_key]
            if old_uid == current:
                continue
            if old_uid != legacy:
                print(f"  Skipping {node.get('name') or node_id} {key}: '{old_uid}' matches neither hash of {directory}; it may have changed since import")
                continue
            renamed[old_uid] = current

    for node_id, node in nodes.items():
        checkpoint, encoder = node.get("checkpoint"), node.get("encoder")
        if not (isinstance(checkpoint, dict) and isinstance(encoder, dict)) or encoder.get("uid") not in renamed:
            continue
        # Model IDs combine the checkpoint and encoder UIDs (see StableAudioAdapter.register_model)
        if node_id != uid_gen.from_uids([checkpoint["uid"], encoder["uid"]]):
            print(f"  Skipping model ID of {node.get('name') or node_id}: it isn't built from its checkpoint and encoder UIDs")
            continue
        new_id = uid_gen.from_uids([renamed.get(checkpoint["uid"], checkpoint["uid"]), renamed[encoder["uid"]]])
        if new_id in nodes:
            print(f"  Skipping model ID of {node.get('name') or node_id}: '{new_id}' is already a node in this graph")
            continue
        renamed[node_id] = new_id
    return renamed


def migrate(project: Path, dry_run: bool, hashed: dict) -> bool:
    graph_file = project / "graph.json"
    with open(graph_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = {
        n["data"]["id"]: n["data"]
        for n in data.get("graph", {}).get("elements", {}).get("nodes", [])
        if "id" in n.get("data", {})
    }
    renamed = plan(nodes, XXH3_64(), hashed)
    if not renamed:
        return False
    for old, new in renamed.items():
        print(f"  {old} -> {new}")
    if dry_run:
        return True

    backup = project / BACKUP_NAME
    if not backup.exists():
        shutil.copy2(graph_file, backup)
    temp = project / f"graph.json.{uuid.uuid4().hex[:8]}.tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(readdress(data, renamed), f, separators=(",", ":"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, graph_file)
    print(f"  Rewrote {graph_file} (original kept at {BACKUP_NAME})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="Project folders, or folders of projects")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing anything")
    args = parser.parse_args()

    projects = []
    for path in args.paths:
        if (path / "graph.json").is_file():
            projects.append(path)
        elif path.is_dir():
            projects += sorted(p.parent for p in path.glob("*/graph.json"))
        else:
            sys.exit(f"Error: {path} is not a project or a folder of projects")

    # Directories shared by several projects are hashed once
    hashed: dict[str, tuple[str, str]] = {}
    changed = 0
    for project in projects:
        print(f"{project.name}:")
        if migrate(project, args.dry_run, hashed):
            changed += 1
        else:
            print("  Nothing to change.")
    verb = "would change" if args.dry_run else "changed"
    print(f"{changed} of {len(projects)} project(s) {verb}.")


if __name__ == "__main__":
    main()
