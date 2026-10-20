"""
Operations that produce several artifacts at once (e.g. a StyleGAN inversion's pivot latent and the
grating tuned around it).

Inside the engine such a result is an OperationResult. Across process boundaries (job status, the
engine service, the remote engine) it travels as a dict:

    {"artifacts": [element dict, ...], "links": [{"source": id, "target": id, "relation": str}, ...]}

The first artifact is the primary one. Single-artifact results stay a plain element dict.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from param_graph.elements.base_elements import GraphElement


@dataclass
class ArtifactLink:
    """A graph edge between two artifacts of the same result."""
    source: str
    target: str
    relation: str

    def to_dict(self) -> dict:
        return {"source": self.source, "target": self.target, "relation": self.relation}


@dataclass
class OperationResult:
    """
    outputs: (artifact, data) pairs. data is the tensor the engine saves as the artifact's file, or None
    when the adapter has already written the file. The first output is the primary artifact.
    links: edges among the outputs. Outputs that are the target of a link are reached through their
    source, so the operation's inputs are linked only to the others (the roots).
    """
    outputs: list[tuple[Any, Any]]
    links: list[ArtifactLink] = field(default_factory=list)


def is_multi_result(result: Any) -> bool:
    return isinstance(result, dict) and isinstance(result.get("artifacts"), list)


def serialize_result(result: Any) -> Any:
    """Converts a job result (GraphElement or multi-artifact dict of GraphElements) to plain dicts."""
    if isinstance(result, GraphElement):
        return result.to_dict()
    if is_multi_result(result):
        return {
            "artifacts": [a.to_dict() if isinstance(a, GraphElement) else a for a in result["artifacts"]],
            "links": [l.to_dict() if isinstance(l, ArtifactLink) else l for l in result.get("links", [])],
        }
    return result


def map_result_elements(result: Any, fn: Callable[[dict], dict]) -> Any:
    """Applies fn to every element dict of a serialized result, single or multi-artifact."""
    if is_multi_result(result):
        return {**result, "artifacts": [fn(a) for a in result["artifacts"]]}
    if isinstance(result, dict) and "id" in result:
        return fn(result)
    return result


def root_ids(result: dict) -> list[str]:
    """IDs of a multi-artifact result's artifacts that no internal link points to, in output order."""
    targets = {link["target"] for link in result.get("links", [])}
    return [a["id"] for a in result["artifacts"] if a["id"] not in targets]


def result_artifacts(result: Any) -> tuple[Optional[dict], list[dict], list[dict]]:
    """Splits a serialized result into (primary element dict, other element dicts, link dicts)."""
    if is_multi_result(result):
        artifacts = result["artifacts"]
        if not artifacts:
            return None, [], []
        return artifacts[0], artifacts[1:], list(result.get("links", []))
    if isinstance(result, dict):
        return result.get("artifact", result) or None, [], []
    return result, [], []
