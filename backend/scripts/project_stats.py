"""Read-only summary statistics for a StemmA2A project.

Usage (from backend/):
    .\\.venv\\Scripts\\python.exe scripts\\project_stats.py <project_dir> [--json]

Reports node and edge counts, rooms and groups, models used, evolution depth and
scoring, grating and genome file sizes, total audio duration, and disk usage.
It never writes anything. Engine compute times are reported for artifacts that
record them (context key "engine_seconds"; projects from before that key existed
have none).
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load_elements(project: Path) -> dict[str, list[dict[str, Any]]]:
    graph_file = project / "graph.json"
    if not graph_file.is_file():
        raise FileNotFoundError(f"No graph.json in {project}")
    doc = json.loads(graph_file.read_text(encoding="utf8"))
    try:
        elements = doc["graph"]["elements"]
        return {"nodes": [n["data"] for n in elements["nodes"]], "edges": [e["data"] for e in elements["edges"]]}
    except (KeyError, TypeError) as error:
        raise ValueError(f"{graph_file} is not a Cytoscape-format project graph: missing {error}") from error


def file_size(node: dict[str, Any], missing: list[str]) -> int | None:
    path = (node.get("file") or {}).get("path")
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        missing.append(path)
        return None
    return p.stat().st_size


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else 0


def size_summary(sizes: list[int]) -> dict[str, float | int] | None:
    if not sizes:
        return None
    return {"count": len(sizes), "median_kb": round(statistics.median(sizes) / 1024, 1),
            "min_kb": round(min(sizes) / 1024, 1), "max_kb": round(max(sizes) / 1024, 1)}


def timing_summary(nodes: list[dict[str, Any]], model_names: dict[str, str | None]) -> dict[str, Any]:
    """Engine compute time per (operation, model), from artifacts that recorded it."""
    times: dict[str, list[float]] = {}
    for n in nodes:
        ctx = n.get("context") or {}
        seconds = ctx.get("engine_seconds")
        if isinstance(seconds, (int, float)):
            key = f"{ctx.get('operation', '?')} · {model_names.get(ctx.get('model_id'), 'unknown')}"
            times.setdefault(key, []).append(float(seconds))
    return {key: {"count": len(v), "median": round(statistics.median(v), 1),
                  "min": round(min(v), 1), "max": round(max(v), 1)} for key, v in times.items()}


def summarize(project: Path) -> dict[str, Any]:
    els = load_elements(project)
    nodes, edges = els["nodes"], els["edges"]
    by_type: dict[str, list[dict[str, Any]]] = {}
    for n in nodes:
        by_type.setdefault(n.get("type", "?"), []).append(n)
    missing: list[str] = []

    individuals = by_type.get("individual", [])
    generations = [i["generation"] for i in individuals if isinstance(i.get("generation"), int)]
    gratings = by_type.get("grating", [])
    kernels = Counter(p.get("kernel_type") for g in gratings for p in g.get("prisms") or [])
    audio = by_type.get("audio", [])
    durations = [a["duration"] for a in audio if isinstance(a.get("duration"), (int, float))]
    model_names = {m["id"]: m.get("name") for m in by_type.get("model", [])}
    used = Counter(model_names.get((n.get("context") or {}).get("model_id"), "unknown")
                   for n in audio + individuals if (n.get("context") or {}).get("model_id"))

    stats: dict[str, Any] = {
        "project": project.name,
        "nodes": len(nodes),
        "nodes_by_type": dict(Counter(n.get("type") for n in nodes)),
        "edges": len(edges),
        "edges_by_type": dict(Counter(e.get("type") for e in edges)),
        "rooms": [r.get("name") for r in by_type.get("room", [])],
        "groups": len(by_type.get("group", [])),
        "models": list(model_names.values()),
        "artifacts_per_model": dict(used),
        "prompts": dict(Counter((n.get("context") or {}).get("prompt") for n in audio + individuals
                                if (n.get("context") or {}).get("prompt")).most_common(10)),
        "evolution": {
            "individuals": len(individuals),
            "max_generation": max(generations) if generations else None,
            "per_generation": dict(sorted(Counter(generations).items())),
            "scored": sum(1 for i in individuals if i.get("fitness") is not None),
        },
        "genome_files": size_summary([s for i in individuals if (s := file_size(i, missing)) is not None]),
        "gratings": {
            "count": len(gratings),
            "kernel_types": dict(kernels),
            "prisms_per_grating": [len(g.get("prisms") or []) for g in gratings],
            "files": size_summary([s for g in gratings if (s := file_size(g, missing)) is not None]),
        },
        "audio": {"count": len(audio), "total_minutes": round(sum(durations) / 60, 1),
                  "median_seconds": round(statistics.median(durations), 1) if durations else None},
        "engine_seconds": timing_summary(nodes, model_names),
        "disk_mb": {d: round(dir_size(project / d) / 2**20, 1) for d in ("cache", "generate", "process")},
        "missing_files": len(missing),
        "edit_conflict_copies": sorted(p.name for p in project.glob("*Edit conflict*")),
    }
    return stats


def print_text(stats: dict[str, Any]) -> None:
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("project", type=Path, help="project folder containing graph.json")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args()
    stats = summarize(args.project.resolve())
    if args.json:
        json.dump(stats, sys.stdout, indent=2)
        print()
    else:
        print_text(stats)


if __name__ == "__main__":
    main()
