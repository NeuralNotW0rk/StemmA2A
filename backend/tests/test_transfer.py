import sys
import shutil
import tempfile
import unittest
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from param_graph.graph import ParameterGraph
from param_graph.transfer import collect_subgraph, write_subgraph

MODEL = "model.xxh3_64"
BASELINE = "grating_base"
GEN0 = "ind0.xxh3_64"
GEN1 = "ind1.xxh3_64"
PRECURSOR = "precursor.xxh3_64"
EXEMPLAR = "exemplar.xxh3_64"
LATENT = "latent.xxh3_64"
MUT_GROUP = "group_mut"


class TestSubgraphTransfer(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.source_root = self.tmp / "source"
        self.target_root = self.tmp / "target"
        self.external = self.tmp / "Models"
        self.external.mkdir()
        self.source = ParameterGraph(str(self.source_root))
        self.source.project_name = "source"
        self._build_lineage()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _file(self, relative: str, uid: str, extension: str, root: Path | None = None) -> dict:
        path = (root or self.source_root) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(uid.encode())
        return {"path": str(path), "uid": uid, "extension": extension, "size": None}

    def _build_lineage(self):
        """model -> baseline grating; precursor audio wrapped as gen 0; gen 1 scattered with an exemplar."""
        G = self.source.G
        G.add_node(MODEL, id=MODEL, type="model", adapter="stable_audio_tools", output_type="audio",
                   name="model", context={}, checkpoint=self._file("sa.ckpt", MODEL, ".ckpt", self.external))
        G.add_node(BASELINE, id=BASELINE, type="grating", name="baseline", base_model_id=MODEL,
                   context={"is_baseline": True}, elements=[],
                   file=self._file("cache/gr/grating_base.safetensors", BASELINE, ".safetensors"))
        G.add_node(LATENT, id=LATENT, type="latent", name="latent", context={},
                   file=self._file("process/latent.pt", LATENT, ".pt"))
        G.add_node(GEN0, id=GEN0, type="individual", name="Baseline", generation=0, base_model_id=MODEL,
                   baseline_grating_id=BASELINE, context={"model_id": MODEL, "precursor_artifact_id": PRECURSOR},
                   file=self._file("generate/ind0.safetensors", GEN0, ".safetensors"))
        G.add_node(PRECURSOR, id=PRECURSOR, type="audio", name="precursor", parent=GEN0,
                   context={"model_id": MODEL, "operation": "generate"},
                   file=self._file("generate/precursor.wav", PRECURSOR, ".wav"),
                   embeddings={"clap": [0.1, 0.2, 0.3]})
        G.add_node(MUT_GROUP, id=MUT_GROUP, type="group", member_ids=[GEN1], member_type="individual")
        G.add_node(GEN1, id=GEN1, type="individual", name="Gen 1", generation=1, base_model_id=MODEL,
                   baseline_grating_id=BASELINE, parent=MUT_GROUP,
                   context={"model_id": MODEL, "precursor_artifact_id": PRECURSOR,
                            "scatter_operation": {"parent_id": GEN0, "mutation_rate": 0.1},
                            "lineage": {"parent_ids": [GEN0], "mutated": True}},
                   file=self._file("generate/ind1.safetensors", GEN1, ".safetensors"))
        G.add_node(EXEMPLAR, id=EXEMPLAR, type="audio", name="exemplar", parent=GEN1, alias="Gen 1 #1",
                   context={"model_id": MODEL, "init_latent_id": LATENT, "operation": "generate",
                            "individuals": [{"id": GEN1, "strength": 1.0}], "baseline_grating_id": BASELINE},
                   file=self._file("generate/exemplar.wav", EXEMPLAR, ".wav"))

        G.add_edge(MODEL, BASELINE, id=f"{MODEL}->{BASELINE}", type="model", relation="binds_to")
        G.add_edge(MODEL, GEN0, id=f"{MODEL}->{GEN0}", type="model", relation="binds_to")
        G.add_edge(MODEL, GEN1, id=f"{MODEL}->{GEN1}", type="model", relation="binds_to")
        G.add_edge(GEN0, GEN1, id=f"{GEN0}->{GEN1}", type="individual", relation="parent")
        G.add_edge(PRECURSOR, EXEMPLAR, id=f"{PRECURSOR}->{EXEMPLAR}", type="spring", spring_type="near")

    def _fork(self, element_ids, include_lineage=True):
        subgraph = collect_subgraph(self.source, element_ids, include_lineage=include_lineage)
        target = ParameterGraph(str(self.target_root))
        target.project_name = "target"
        report = write_subgraph(subgraph, target)
        target.save()
        reloaded = ParameterGraph(str(self.target_root))
        self.assertTrue(reloaded.load())
        return reloaded, report

    def test_individual_brings_exemplars_lineage_and_requirements(self):
        target, report = self._fork([GEN1])
        G = target.G

        self.assertEqual(set(G.nodes), {GEN1, EXEMPLAR, GEN0, PRECURSOR, MODEL, BASELINE})
        self.assertEqual(report["converted_exemplars"], [])
        # The exemplar stays inside its individual; the scatter group was left behind
        self.assertEqual(G.nodes[EXEMPLAR]["parent"], GEN1)
        self.assertNotIn("parent", G.nodes[GEN1])
        self.assertTrue(G.has_edge(GEN0, GEN1))
        self.assertFalse(any(d.get("type") == "spring" for _, _, d in G.edges(data=True)))

        # Project files are copied to the same relative place; the external model stays in place
        self.assertEqual(Path(G.nodes[EXEMPLAR]["file"]["path"]), self.target_root / "generate" / "exemplar.wav")
        self.assertEqual(Path(G.nodes[BASELINE]["file"]["path"]),
                         self.target_root / "cache" / "gr" / "grating_base.safetensors")
        for node_id in (EXEMPLAR, GEN0, GEN1, BASELINE, PRECURSOR):
            self.assertTrue(Path(G.nodes[node_id]["file"]["path"]).exists())
        self.assertEqual(Path(G.nodes[MODEL]["checkpoint"]["path"]), self.external / "sa.ckpt")

        # Embeddings travel through the sidecar; the source project is untouched
        for stored, expected in zip(G.nodes[PRECURSOR]["embeddings"]["clap"], [0.1, 0.2, 0.3]):
            self.assertAlmostEqual(stored, expected, places=6)
        self.assertEqual(Path(self.source.G.nodes[EXEMPLAR]["file"]["path"]).parent, self.source_root / "generate")
        self.assertEqual(self.source.G.nodes[GEN1]["parent"], MUT_GROUP)

    def test_exemplar_without_individual_becomes_plain_audio(self):
        target, report = self._fork([EXEMPLAR])
        G = target.G

        self.assertEqual(report["converted_exemplars"], [EXEMPLAR])
        self.assertNotIn(GEN1, G)
        self.assertNotIn(GEN0, G)
        # Its own inputs are followed once the individual no longer implies them
        self.assertIn(MODEL, G)
        self.assertIn(LATENT, G)

        attrs = G.nodes[EXEMPLAR]
        self.assertNotIn("parent", attrs)
        self.assertNotIn("alias", attrs)
        self.assertNotIn("individuals", attrs["context"])
        self.assertFalse(target.is_generated_exemplar(EXEMPLAR))

        ancestry = attrs["context"]["ancestry"]
        self.assertEqual(ancestry["individuals"], [{"id": GEN1, "strength": 1.0}])
        self.assertEqual(ancestry["baseline_grating_id"], BASELINE)
        self.assertEqual([a["id"] for a in ancestry["lineage"]], [GEN1, GEN0])
        self.assertEqual(ancestry["lineage"][0]["scatter_operation"]["parent_id"], GEN0)
        self.assertEqual(ancestry["lineage"][1]["precursor_artifact_id"], PRECURSOR)
        # The source keeps the exemplar as it was
        self.assertIn("individuals", self.source.G.nodes[EXEMPLAR]["context"])

    def test_without_lineage_only_requirements_come_along(self):
        target, _ = self._fork([GEN1], include_lineage=False)
        self.assertEqual(set(target.G.nodes), {GEN1, EXEMPLAR, MODEL, BASELINE})

    def test_missing_project_file_fails_before_copying(self):
        Path(self.source.G.nodes[GEN0]["file"]["path"]).unlink()
        subgraph = collect_subgraph(self.source, [GEN1])
        target = ParameterGraph(str(self.target_root))
        with self.assertRaises(FileNotFoundError):
            write_subgraph(subgraph, target)
        self.assertFalse(self.target_root.exists())

    def test_missing_file_falls_back_to_cas_copy(self):
        path = Path(self.source.G.nodes[LATENT]["file"]["path"])
        cas = self.source_root / "cache" / "la" / "latent.pt"
        cas.parent.mkdir(parents=True)
        path.rename(cas)
        target, _ = self._fork([LATENT])
        self.assertEqual(Path(target.G.nodes[LATENT]["file"]["path"]), self.target_root / "cache" / "la" / "latent.pt")

    def test_unknown_element_is_rejected(self):
        with self.assertRaises(ValueError):
            collect_subgraph(self.source, ["nope"])


if __name__ == "__main__":
    unittest.main()
