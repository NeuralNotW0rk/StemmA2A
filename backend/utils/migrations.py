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

    def referenced_baseline(ctx):
        """Returns the baseline grating of the first individual an artifact context was generated from."""
        individual_ids = list(ctx.get("individual_ids") or [])
        individual_ids += [i.get("id") for i in ctx.get("individuals") or [] if isinstance(i, dict)]
        for ind_id in individual_ids:
            grating_id = (node_data.get(ind_id) or {}).get("baseline_grating_id")
            if grating_id in node_data:
                return grating_id
        return None

    modified = False
    # Individuals (and copies of their contexts) first, so artifacts can then follow them to their baseline
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

    for d in list(node_data.values()):
        ctx = d.get("context")
        serialized = ctx.get("baseline_grating") if isinstance(ctx, dict) else None
        if isinstance(serialized, dict):
            # Serialized copies may carry paths anchored on a remote engine, so they are matched to
            # an existing grating (by ID, source individual, or local path) and never create one
            grating_id = serialized.get("id")
            if grating_id not in node_data:
                grating_id = referenced_baseline(ctx)
            if grating_id is None:
                file_info = serialized.get("file") if isinstance(serialized.get("file"), dict) else {}
                grating_id = grating_by_path.get(normalize(file_info["path"])) if file_info.get("path") else None
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


def migrate_genome_encoding(project_path: Path) -> None:
    """
    Converts unversioned LoRAGenome files, which store phenotype weights, to the current format, which
    stores genotype values decoded with scales from their shapes (see lora_gene_scales), plus each gene's
    topology, taken from the individual's baseline grating.

    A genome's uid is the hash of its content, so each converted genome is re-addressed: it is saved under
    the uid of its new content, and every reference to the old uid in graph.json (node IDs, edges, group
    members, lineage) is rewritten. Each conversion is checked to decode back to its original weights first.
    The new files are written before graph.json (whose original is kept as graph.json.pre-genome-migration),
    and the old files are only removed after it, so an interrupted run leaves a loadable project.
    Individuals whose baseline grating cannot be found are left unconverted and reported.
    """
    if not project_path or not project_path.is_dir():
        return

    graph_file = project_path / "graph.json"
    if not graph_file.exists() or graph_file.stat().st_size == 0:
        return

    import json
    import shutil
    import uuid

    try:
        with open(graph_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse {graph_file} for migration (invalid JSON): {e}")
        return

    nodes = data.get("graph", {}).get("elements", {}).get("nodes", [])
    node_data = {node["data"]["id"]: node["data"] for node in nodes if "id" in node.get("data", {})}
    individuals = [d for d in node_data.values() if d.get("type") == "individual"]
    if not individuals:
        return

    generate_dir = project_path / "generate"

    def resolve_file(d: dict) -> Path | None:
        """
        Finds a node's file, preferring the project's own generate folder so a copied or moved project
        never rewrites the original's files, and falling back to the stored path.
        """
        file_info = d.get("file") if isinstance(d.get("file"), dict) else {}
        candidates = []
        if file_info.get("uid"):
            candidates.append(generate_dir / f"{file_info['uid']}{file_info.get('extension') or '.safetensors'}")
        candidates.append(file_info.get("path"))
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                return Path(candidate)
        return None

    from safetensors import safe_open
    from diffracture.topology.grating import Grating as DiffractureGrating
    from evolution.lora_genome import GENOME_FORMAT, LoRAGene, LoRAGenome, lora_gene_scales
    from utils.uid import XXH3_64

    uid_gen = XXH3_64()
    baseline_topology: dict[Path, dict[str, dict]] = {}
    seen: set[Path] = set()
    readdressed: dict[str, str] = {}
    new_paths: dict[str, Path] = {}
    old_files: list[Path] = []
    unresolved: list[str] = []

    for ind in individuals:
        genome_path = resolve_file(ind)
        old_uid = (ind.get("file") or {}).get("uid") or ind["id"]
        if genome_path is None or genome_path in seen:
            continue
        seen.add(genome_path)

        with safe_open(str(genome_path), framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            if metadata.get("format") is not None or "addresses" not in metadata:
                continue
            tensors = {key: f.get_tensor(key) for key in f.keys()}

        # Legacy files hold no topology; it comes from the baseline the individual was bred from
        baseline_path = resolve_file(node_data.get(ind.get("baseline_grating_id")) or {})
        if baseline_path is None:
            unresolved.append(ind.get("name") or ind["id"])
            continue
        if baseline_path not in baseline_topology:
            baseline = DiffractureGrating.load(str(baseline_path))
            baseline_topology[baseline_path] = {
                address: LoRAGene.from_element(el).topology
                for address, el in baseline.nodes.items()
                if "lora_down" in el.params and "lora_up" in el.params
            }
        topologies = baseline_topology[baseline_path]

        gene_metadata = json.loads(metadata["gene_metadata"])
        genes = []
        for address in json.loads(metadata["addresses"]):
            weights = (tensors[f"{address}.lora_down"], tensors[f"{address}.lora_up"])
            scales = lora_gene_scales(*weights)
            encoded = [w / scale for w, scale in zip(weights, scales)]
            for name, value, scale, ref in zip(("lora_down", "lora_up"), encoded, scales, weights):
                decoded_error = (value * scale - ref).abs().max().item()
                if decoded_error > 1e-5 * max(1.0, ref.abs().max().item()):
                    raise RuntimeError(f"Genome {genome_path.name} does not round-trip at {address}.{name} (error {decoded_error}).")
            genes.append(LoRAGene(
                address=address,
                lora_down=encoded[0].contiguous(),
                lora_up=encoded[1].contiguous(),
                activation=bool(gene_metadata[address]["active"]),
                topology=topologies.get(address, {}),
            ))
        genome = LoRAGenome(genes)

        generate_dir.mkdir(exist_ok=True)
        temp_path = generate_dir / f"genome_migration.{uuid.uuid4().hex[:8]}.tmp"
        genome.save(temp_path)
        # The same hash every operation uses when it mints an individual's uid
        new_uid = uid_gen.from_state_dict(genome.get_state_dict())
        if new_uid != old_uid and (new_uid in node_data or new_uid in new_paths):
            temp_path.unlink()
            print(f"Warning: converted genome '{old_uid}' matches existing node '{new_uid}'; left unconverted.")
            continue

        new_path = generate_dir / f"{new_uid}.safetensors"
        os.replace(temp_path, new_path)
        readdressed[old_uid] = new_uid
        new_paths[new_uid] = new_path
        if genome_path != new_path and genome_path.parent == generate_dir:
            old_files.append(genome_path)

    if readdressed:
        def swap(value):
            """Replaces old uids wherever they appear whole, including either end of an edge ID like a->b."""
            if isinstance(value, str):
                if value in readdressed:
                    return readdressed[value]
                if "->" in value:
                    return "->".join(readdressed.get(part, part) for part in value.split("->"))
                return value
            if isinstance(value, list):
                return [swap(v) for v in value]
            if isinstance(value, dict):
                return {swap(k): swap(v) for k, v in value.items()}
            return value

        data = swap(data)
        for node in data.get("graph", {}).get("elements", {}).get("nodes", []):
            file_info = node.get("data", {}).get("file")
            if isinstance(file_info, dict) and file_info.get("uid") in new_paths:
                file_info["path"] = str(new_paths[file_info["uid"]])

        backup_file = project_path / "graph.json.pre-genome-migration"
        if not backup_file.exists():
            shutil.copy2(graph_file, backup_file)
        temp_graph = project_path / f"graph.json.{uuid.uuid4().hex[:8]}.tmp"
        with open(temp_graph, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_graph, graph_file)

        for old_file in old_files:
            try:
                old_file.unlink()
            except OSError as e:
                print(f"Warning: could not remove migrated genome file {old_file.name}: {e}")

        print(f"Converted and re-addressed {len(readdressed)} genome(s) in {project_path.name} to genotype format {GENOME_FORMAT} (original graph kept at {backup_file.name})")
    if unresolved:
        print(f"Warning: {len(unresolved)} genome(s) in {project_path.name} have no baseline grating to take topology from and were not converted: {', '.join(unresolved[:5])}")


def migrate_prune_baselines(project_path: Path) -> None:
    """
    Removes hidden baseline gratings (context is_baseline) that only served as templates for expressing
    genomes. Genomes now carry their own topology, so a baseline is removed when every element in it is a
    LoRA element (all of it lives in the genomes) and no individual referencing it still has an unconverted
    legacy genome. References to it are cleared and its edges dropped; graph.json is rewritten (original kept
    as graph.json.pre-baseline-pruning) before the baseline files are deleted.
    """
    if not project_path or not project_path.is_dir():
        return

    graph_file = project_path / "graph.json"
    if not graph_file.exists() or graph_file.stat().st_size == 0:
        return

    import json
    import shutil
    import uuid

    try:
        with open(graph_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse {graph_file} for migration (invalid JSON): {e}")
        return

    elements = data.get("graph", {}).get("elements", {})
    nodes = elements.get("nodes", [])
    node_data = {node["data"]["id"]: node["data"] for node in nodes if "id" in node.get("data", {})}
    baselines = [
        d for d in node_data.values()
        if d.get("type") == "grating" and isinstance(d.get("context"), dict) and d["context"].get("is_baseline")
    ]
    if not baselines:
        return

    from safetensors import safe_open

    generate_dir = project_path / "generate"

    def resolve_file(d: dict) -> Path | None:
        """Finds a node's file, preferring the project's own generate folder, then the stored path."""
        file_info = d.get("file") if isinstance(d.get("file"), dict) else {}
        candidates = []
        if file_info.get("uid"):
            candidates.append(generate_dir / f"{file_info['uid']}{file_info.get('extension') or '.safetensors'}")
        candidates.append(file_info.get("path"))
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                return Path(candidate)
        return None

    def kernel_types(d: dict, path: Path | None) -> list:
        if path is not None:
            with safe_open(str(path), framework="pt", device="cpu") as f:
                topology = (f.metadata() or {}).get("topology")
            if topology:
                return [node.get("kernel_type") for node in json.loads(topology)]
        return [el.get("kernel_type") for el in d.get("elements") or []]

    def is_legacy_genome(d: dict) -> bool:
        path = resolve_file(d)
        if path is None:
            return False
        with safe_open(str(path), framework="pt", device="cpu") as f:
            return (f.metadata() or {}).get("format") is None

    removable: dict[str, Path | None] = {}
    for baseline in baselines:
        path = resolve_file(baseline)
        if any(kind != "lora" for kind in kernel_types(baseline, path)):
            continue
        dependants = [
            d for d in node_data.values()
            if d.get("type") == "individual" and d.get("baseline_grating_id") == baseline["id"]
        ]
        if any(is_legacy_genome(d) for d in dependants):
            continue
        removable[baseline["id"]] = path

    if not removable:
        return

    def prune(value):
        """Drops removed IDs from lists and clears them where they appear as a value."""
        if isinstance(value, list):
            return [prune(v) for v in value if not (isinstance(v, str) and v in removable)]
        if isinstance(value, dict):
            return {k: (None if isinstance(v, str) and v in removable else prune(v)) for k, v in value.items()}
        return value

    elements["nodes"] = [n for n in nodes if n.get("data", {}).get("id") not in removable]
    elements["edges"] = [
        e for e in elements.get("edges", [])
        if e.get("data", {}).get("source") not in removable and e.get("data", {}).get("target") not in removable
    ]
    data = prune(data)

    backup_file = project_path / "graph.json.pre-baseline-pruning"
    if not backup_file.exists():
        shutil.copy2(graph_file, backup_file)
    temp_graph = project_path / f"graph.json.{uuid.uuid4().hex[:8]}.tmp"
    with open(temp_graph, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_graph, graph_file)

    # Only files inside this project are deleted, and only when no remaining node still uses them
    still_used = {resolve_file(d) for d in node_data.values() if d.get("id") not in removable}
    for path in removable.values():
        if path is not None and path.parent == generate_dir and path not in still_used:
            try:
                path.unlink()
            except OSError as e:
                print(f"Warning: could not remove baseline grating file {path.name}: {e}")

    print(f"Removed {len(removable)} baseline grating(s) no longer needed in {project_path.name} (original graph kept at {backup_file.name})")


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
    migrate_genome_encoding(project_path)
    migrate_prune_baselines(project_path)
