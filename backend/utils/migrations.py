import os
from pathlib import Path

# Feature flag to enable/disable all filesystem/compatibility migrations
ENABLE_MIGRATIONS = os.environ.get("ENABLE_MIGRATIONS", "true").lower() == "true"


def migrate_sharded_cache(root_dir: Path) -> None:
    """
    Migrates any old-style 2-character sharded cache folders
    from root_dir to root_dir / 'cache'.
    """
    import shutil
    import re

    if not root_dir or not root_dir.is_dir():
        return
    
    cache_dir = root_dir / "cache"
    hex_pattern = re.compile(r"^[0-9a-fA-F]{2}$")
    
    try:
        for path in root_dir.iterdir():
            if path.is_dir() and hex_pattern.match(path.name):
                dest = cache_dir / path.name
                cache_dir.mkdir(exist_ok=True)
                
                if dest.exists():
                    for item in path.iterdir():
                        item_dest = dest / item.name
                        if not item_dest.exists():
                            shutil.move(str(item), item_dest)
                    try:
                        path.rmdir()
                    except OSError:
                        pass
                else:
                    shutil.move(str(path), dest)
                print(f"Migrated sharded cache folder {path.name} to {dest}")
    except Exception as e:
        print(f"Failed to migrate sharded cache in {root_dir}: {e}")


def migrate_batch_to_group(project_path: Path) -> None:
    """
    Migrates any legacy 'batch' elements in graph.json to 'group' elements.
    """
    if not project_path or not project_path.is_dir():
        return

    graph_file = project_path / "graph.json"
    if not graph_file.exists() or graph_file.stat().st_size == 0:
        return

    import json
    try:
        with open(graph_file, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse {graph_file} for migration (invalid JSON): {e}")
        return

    modified = False
    
    graph_data = data.get("graph", {})
    elements = graph_data.get("elements", {})
    
    nodes = elements.get("nodes", [])
    for node in nodes:
        node_data = node.get("data", {})
        if node_data.get("type") == "batch":
            node_data["type"] = "group"
            modified = True

    edges = elements.get("edges", [])
    for edge in edges:
        edge_data = edge.get("data", {})
        if edge_data.get("type") == "batch":
            edge_data["type"] = "group"
            modified = True

    if modified:
        with open(graph_file, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"Migrated legacy batch nodes to group nodes in {graph_file}")


def migrate_baseline_references(project_path: Path) -> None:
    """
    Replaces per-node copies of the evolution baseline with references to a single baseline Grating:
    - Nodes whose context stores baseline_elements / baseline_file_path (individuals, and artifacts
      whose context was copied from one) point at a Grating node instead, via baseline_grating_id.
      A hidden (is_baseline) Grating node is created for each baseline file not already in the graph.
    - Contexts holding a whole serialized baseline_grating element keep only its ID.
    The original file is kept as graph.json.pre-baseline-migration before anything is rewritten.
    """
    if not project_path or not project_path.is_dir():
        return

    graph_file = project_path / "graph.json"
    if not graph_file.exists() or graph_file.stat().st_size == 0:
        return

    import json
    import shutil
    import uuid
    from param_graph.elements.artifacts.grating_element import Grating
    from param_graph.elements.base_elements import Asset

    try:
        with open(graph_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse {graph_file} for migration (invalid JSON): {e}")
        return

    elements = data.get("graph", {}).get("elements", {})
    nodes = elements.setdefault("nodes", [])
    edges = elements.setdefault("edges", [])
    node_data = {node["data"]["id"]: node["data"] for node in nodes if "id" in node.get("data", {})}

    normalize = lambda p: os.path.normcase(os.path.abspath(str(p)))
    grating_by_path = {
        normalize(d["file"]["path"]): node_id
        for node_id, d in node_data.items()
        if d.get("type") == "grating" and isinstance(d.get("file"), dict) and d["file"].get("path")
    }

    def resolve_baseline(file_path, baseline_elements, base_model_id):
        """Returns the Grating node ID for a baseline file, creating a hidden baseline node if needed."""
        if not file_path:
            return None
        key = normalize(file_path)
        if key not in grating_by_path:
            grating_id = f"grating_{uuid.uuid4().hex[:8]}"
            grating = Grating(
                id=grating_id,
                name="Baseline Grating",
                context={"is_baseline": True},
                file=Asset(path=str(file_path), uid=grating_id, extension=".safetensors"),
                base_model_id=base_model_id,
                elements=baseline_elements or [],
            )
            attrs = grating.to_dict()
            nodes.append({"data": {**attrs, "value": grating_id}})
            node_data[grating_id] = attrs
            model = node_data.get(base_model_id)
            if model:
                edges.append({"data": {
                    "type": model.get("type"),
                    "id": f"{base_model_id}->{grating_id}",
                    "relation": "binds_to",
                    "source": base_model_id,
                    "target": grating_id,
                }})
            grating_by_path[key] = grating_id
        return grating_by_path[key]

    modified = False
    for d in list(node_data.values()):
        ctx = d.get("context")
        if not isinstance(ctx, dict):
            continue

        if "baseline_elements" in ctx or "baseline_file_path" in ctx:
            grating_id = d.get("baseline_grating_id") or ctx.get("baseline_grating_id")
            if grating_id not in node_data:
                grating_id = resolve_baseline(
                    ctx.get("baseline_file_path"),
                    ctx.get("baseline_elements"),
                    d.get("base_model_id") or ctx.get("model_id"),
                )
            if grating_id:
                ctx.pop("baseline_elements", None)
                ctx.pop("baseline_file_path", None)
                if "baseline_grating_id" in ctx:
                    ctx["baseline_grating_id"] = grating_id
                if d.get("type") == "individual":
                    d["baseline_grating_id"] = grating_id
                modified = True

        serialized = ctx.get("baseline_grating")
        if isinstance(serialized, dict):
            grating_id = serialized.get("id")
            if grating_id not in node_data:
                file_info = serialized.get("file") if isinstance(serialized.get("file"), dict) else {}
                grating_id = resolve_baseline(
                    file_info.get("path"),
                    serialized.get("elements"),
                    serialized.get("base_model_id"),
                )
            del ctx["baseline_grating"]
            if grating_id:
                ctx["baseline_grating_id"] = grating_id
            modified = True

    if not modified:
        return

    backup_file = project_path / "graph.json.pre-baseline-migration"
    if not backup_file.exists():
        shutil.copy2(graph_file, backup_file)

    temp_file = project_path / f"graph.json.{uuid.uuid4().hex[:8]}.tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_file, graph_file)
    print(f"Replaced copied baseline gratings with references in {graph_file} (original kept at {backup_file.name})")


def run_global_migrations(data_cache_root: Path) -> None:
    """
    Runs all startup global data cache migrations.
    """
    if not ENABLE_MIGRATIONS:
        return
    
    print("Running global migrations...")
    migrate_sharded_cache(data_cache_root)


def run_project_migrations(project_path: Path) -> None:
    """
    Runs all project-level migrations when a project is loaded.
    """
    if not ENABLE_MIGRATIONS:
        return
    
    print(f"Running project migrations for {project_path.name}...")
    migrate_sharded_cache(project_path)
    migrate_batch_to_group(project_path)
    migrate_baseline_references(project_path)
