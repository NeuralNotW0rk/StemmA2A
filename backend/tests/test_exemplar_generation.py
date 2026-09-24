import unittest
import torch
import tempfile
import shutil
import os
import time
import uuid
from unittest.mock import AsyncMock

from param_graph.graph import ParameterGraph
from param_graph.elements.models.stylegan_element import StyleGANModel
from param_graph.elements.artifacts.individual_element import Individual
from param_graph.elements.artifacts.audio_element import Audio
from param_graph.elements.artifacts.grating_element import Grating
from param_graph.elements.collections.group_element import Group
from param_graph.elements.base_elements import Asset
from evolution.lora_genome import LoRAGene, LoRAGenome
from engine.engine_provider import EngineProvider
import app as app_module


class TestExemplarGeneration(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.tmp_dir = tempfile.mkdtemp()
        self.graph = ParameterGraph(self.tmp_dir)
        self.provider = EngineProvider(data_root=self.tmp_dir)

        self.mock_engine = self.provider.get_engine()
        self.mock_engine.execute = AsyncMock(
            side_effect=lambda *args, **kwargs: kwargs.get("job_id") or str(uuid.uuid4())
        )

        async def _mock_get_job_status(jid):
            fpath = os.path.join(self.tmp_dir, f"mock_audio_{uuid.uuid4().hex[:6]}.wav")
            with open(fpath, "wb") as f:
                f.write(b"RIFFmockwavdata")
            return {
                "status": "completed",
                "result": {
                    "artifact": {
                        "type": "audio",
                        "id": f"audio_child_{uuid.uuid4().hex[:6]}",
                        "name": "Generated Exemplar Audio",
                        "file": {"path": fpath, "uid": f"uid_{uuid.uuid4().hex[:6]}", "extension": ".wav"},
                        "context": {}
                    }
                }
            }

        self.mock_engine.get_job_status = AsyncMock(side_effect=_mock_get_job_status)

        self.old_graph = app_module.param_graph
        self.old_provider = app_module.engine_provider
        self.old_trigger = app_module.trigger_embedding_update
        app_module.param_graph = self.graph
        app_module.engine_provider = self.provider
        app_module.trigger_embedding_update = AsyncMock(return_value=None) if hasattr(app_module.trigger_embedding_update, "__await__") else unittest.mock.MagicMock()

        self.client = app_module.app.test_client()

    def tearDown(self) -> None:
        app_module.param_graph = self.old_graph
        app_module.engine_provider = self.old_provider
        app_module.trigger_embedding_update = self.old_trigger
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_decoupled_recombination_offline(self) -> None:
        """Verify that recombination completes purely offline on CPU without calling engine.execute."""
        # 1. Model
        model_node = StyleGANModel(
            id="model_recomb_offline",
            name="Offline Recomb Model",
            context={},
            checkpoint=Asset(path="mock_model_path", uid="mock_model_uid", extension=".pt"),
            adapter="stylegan2"
        )
        self.graph.add_element(model_node)

        # 2. Baseline Grating
        grating_node = Grating(
            id="grating_baseline_1",
            name="Baseline Grating",
            context={},
            file=Asset(path=os.path.join(self.tmp_dir, "base_grating.safetensors"), uid="base_grating_uid", extension=".safetensors"),
            base_model_id="model_recomb_offline",
            elements=[]
        )
        self.graph.add_element(grating_node)

        # 3. Gen 0 Parent Individuals
        output_dir = self.graph.root / "generate"
        os.makedirs(output_dir, exist_ok=True)
        parent_ids = []
        for i in range(2):
            gene = LoRAGene(f"layer_{i}", torch.randn(2, 2), torch.randn(2, 2), True)
            genome = LoRAGenome([gene])
            gpath = output_dir / f"p_{i}.safetensors"
            genome.save(str(gpath))

            ind = Individual(
                id=f"p_ind_{i}",
                name=f"Parent {i}",
                file=Asset(path=str(gpath), uid=f"p_ind_{i}", extension=".safetensors"),
                base_model_id="model_recomb_offline",
                baseline_grating_id="grating_baseline_1",
                generation=0,
                fitness=float(i * 10),
                context={"model_id": "model_recomb_offline", "baseline_elements": []}
            )
            self.graph.add_element(ind)
            self.graph.link(model_node, ind, relation="binds_to")
            parent_ids.append(ind.id)

        self.graph.save()

        # Recombination should NOT touch engine.execute
        self.mock_engine.execute.reset_mock()

        resp = self.client.post("/recombine_evolution", json={
            "parent_ids": parent_ids,
            "offspring_size": 3,
            "selection_type": "tournament",
            "crossover_type": "two_point",
            "mutation_rate": 0.05
        })
        self.assertEqual(resp.status_code, 202)
        job_id = resp.get_json()["job_id"]

        completed_data = None
        for _ in range(50):
            status_resp = self.client.get(f"/job_status/{job_id}")
            if status_resp.status_code == 200:
                s_data = status_resp.get_json()
                if s_data.get("status") == "completed":
                    completed_data = s_data.get("result")
                    break
                elif s_data.get("status") == "failed":
                    self.fail(f"Recombination failed: {s_data.get('error')}")
            time.sleep(0.01)

        self.assertIsNotNone(completed_data)
        self.assertEqual(len(completed_data["individual_ids"]), 3)
        self.assertEqual(completed_data["generation"], 1)

        # Engine execute should NOT have been called during recombination
        self.mock_engine.execute.assert_not_called()

        # Parameter graph contains generation group and 3 new individual nodes
        self.graph.load()
        gen_group = self.graph.get_element(completed_data["group_id"])
        self.assertIsInstance(gen_group, Group)
        self.assertEqual(len(gen_group.member_ids), 3)

    def test_decoupled_mutation_offline(self) -> None:
        """Verify that mutation completes purely offline on CPU without calling engine.execute."""
        model_node = StyleGANModel(
            id="model_mutate_offline",
            name="Offline Mutate Model",
            context={},
            checkpoint=Asset(path="mock_model_path", uid="mock_model_uid", extension=".pt"),
            adapter="stylegan2"
        )
        self.graph.add_element(model_node)

        output_dir = self.graph.root / "generate"
        os.makedirs(output_dir, exist_ok=True)
        gene = LoRAGene("layer_mut", torch.randn(2, 2), torch.randn(2, 2), True)
        genome = LoRAGenome([gene])
        gpath = output_dir / "parent_mut.safetensors"
        genome.save(str(gpath))

        parent_ind = Individual(
            id="parent_mut_1",
            name="Parent Mut",
            file=Asset(path=str(gpath), uid="parent_mut_1", extension=".safetensors"),
            base_model_id="model_mutate_offline",
            generation=0,
            context={"model_id": "model_mutate_offline", "baseline_elements": []}
        )
        self.graph.add_element(parent_ind)
        self.graph.link(model_node, parent_ind, relation="binds_to")
        self.graph.save()

        self.mock_engine.execute.reset_mock()

        resp = self.client.post("/execute_operation", json={
            "operation": "mutate",
            "parents": [parent_ind.id],
            "offspring_size": 2,
            "lora_noise": 0.05,
            "mutation_rate": 0.5
        })
        self.assertEqual(resp.status_code, 202)
        job_id = resp.get_json()["job_id"]

        completed_data = None
        for _ in range(50):
            status_resp = self.client.get(f"/job_status/{job_id}")
            if status_resp.status_code == 200:
                s_data = status_resp.get_json()
                if s_data.get("status") == "completed":
                    completed_data = s_data.get("result")
                    break
                elif s_data.get("status") == "failed":
                    self.fail(f"Mutation failed: {s_data.get('error')}")
            time.sleep(0.01)

        self.assertIsNotNone(completed_data)
        self.assertEqual(len(completed_data["individual_ids"]), 2)
        self.mock_engine.execute.assert_not_called()

    def test_generate_exemplar_for_individual(self) -> None:
        """Verify that triggering generate on an Individual routes to engine.execute and parents the audio node to the individual."""
        model_node = StyleGANModel(
            id="model_exemplar_test",
            name="Exemplar Test Model",
            context={},
            checkpoint=Asset(path="mock_model_path", uid="mock_model_uid", extension=".pt"),
            adapter="stylegan2"
        )
        self.graph.add_element(model_node)

        output_dir = self.graph.root / "generate"
        os.makedirs(output_dir, exist_ok=True)
        gene = LoRAGene("layer_ex", torch.randn(2, 2), torch.randn(2, 2), True)
        genome = LoRAGenome([gene])
        gpath = output_dir / "target_ind.safetensors"
        genome.save(str(gpath))

        ind = Individual(
            id="ind_for_exemplar",
            name="Individual For Exemplar",
            file=Asset(path=str(gpath), uid="ind_for_exemplar", extension=".safetensors"),
            base_model_id="model_exemplar_test",
            generation=1,
            context={"model_id": "model_exemplar_test", "baseline_elements": []}
        )
        self.graph.add_element(ind)
        self.graph.link(model_node, ind, relation="binds_to")
        self.graph.save()

        # Dispatch generate exemplar with initiator as Individual
        gen_job_id = f"job_exemplar_{uuid.uuid4().hex[:8]}"
        resp = self.client.post("/execute_operation", json={
            "job_id": gen_job_id,
            "operation": "generate",
            "initiator": ind.to_dict(),
            "params": {
                "truncation": 0.8
            }
        })
        self.assertEqual(resp.status_code, 202)

        # Verify engine.execute was called with individual_elements containing ind
        self.mock_engine.execute.assert_called_once()
        call_args, call_kwargs = self.mock_engine.execute.call_args
        self.assertEqual(call_args[0], "generate")
        self.assertIn("individual_elements", call_kwargs)
        self.assertEqual(call_kwargs["individual_elements"][0].id, ind.id)

        # Poll job status to complete artifact registration
        status_resp = self.client.get(f"/job_status/{gen_job_id}")
        self.assertEqual(status_resp.status_code, 200)
        res = status_resp.get_json()
        self.assertEqual(res["status"], "completed")
        audio_id = res["node_id"]

        # Verify graph linking: Audio node is parented to Individual
        self.graph.load()
        audio_elem = self.graph.get_element(audio_id)
        self.assertIsNotNone(audio_elem)
        self.assertEqual(self.graph.G.nodes[audio_id].get("parent"), ind.id)

    def test_recombine_with_auto_exemplar_generation(self) -> None:
        """Verify that recombination with generate_exemplars=True queues engine jobs and parents exemplars to individuals."""
        model_node = StyleGANModel(
            id="model_recomb_auto_ex",
            name="Auto Ex Recomb Model",
            context={},
            checkpoint=Asset(path="mock_model_path", uid="mock_model_uid", extension=".pt"),
            adapter="stylegan2"
        )
        self.graph.add_element(model_node)

        output_dir = self.graph.root / "generate"
        os.makedirs(output_dir, exist_ok=True)
        parent_ids = []
        for i in range(2):
            gene = LoRAGene(f"layer_auto_{i}", torch.randn(2, 2), torch.randn(2, 2), True)
            genome = LoRAGenome([gene])
            gpath = output_dir / f"p_auto_{i}.safetensors"
            genome.save(str(gpath))

            ind = Individual(
                id=f"p_auto_ind_{i}",
                name=f"Parent Auto {i}",
                file=Asset(path=str(gpath), uid=f"p_auto_ind_{i}", extension=".safetensors"),
                base_model_id="model_recomb_auto_ex",
                generation=0,
                fitness=float(i * 10),
                context={"model_id": "model_recomb_auto_ex", "baseline_elements": []}
            )
            self.graph.add_element(ind)
            self.graph.link(model_node, ind, relation="binds_to")
            parent_ids.append(ind.id)

        self.graph.save()
        self.mock_engine.execute.reset_mock()

        resp = self.client.post("/recombine_evolution", json={
            "parent_ids": parent_ids,
            "offspring_size": 2,
            "selection_type": "tournament",
            "crossover_type": "two_point",
            "generate_exemplars": True,
            "params": {
                "truncation": 0.7
            }
        })
        self.assertEqual(resp.status_code, 202)
        job_id = resp.get_json()["job_id"]

        completed_data = None
        for _ in range(50):
            status_resp = self.client.get(f"/job_status/{job_id}")
            if status_resp.status_code == 200:
                s_data = status_resp.get_json()
                if s_data.get("status") == "completed":
                    completed_data = s_data.get("result")
                    break
                elif s_data.get("status") == "failed":
                    self.fail(f"Recombination failed: {s_data.get('error')}")
            time.sleep(0.01)

        self.assertIsNotNone(completed_data)
        self.assertEqual(len(completed_data["individual_ids"]), 2)
        self.assertIn("job_ids", completed_data)
        self.assertEqual(len(completed_data["job_ids"]), 2)

        # Verify engine.execute was called twice (once per child individual)
        self.assertEqual(self.mock_engine.execute.call_count, 2)

        # Poll and complete each exemplar sub-job
        for sub_job_id in completed_data["job_ids"]:
            sub_resp = self.client.get(f"/job_status/{sub_job_id}")
            self.assertEqual(sub_resp.status_code, 200)
            sub_res = sub_resp.get_json()
            self.assertEqual(sub_res["status"], "completed")

        # Reload graph and verify that each child individual has an exemplar audio parented to it
        self.graph.load()
        for child_id in completed_data["individual_ids"]:
            child_node = self.graph.get_element(child_id)
            self.assertIsNotNone(child_node)
            # Find audio artifact parented to this child individual
            exemplar_audios = [
                node_id for node_id, attrs in self.graph.G.nodes(data=True)
                if attrs.get("parent") == child_id and self.graph.get_element(node_id).type == "audio"
            ]
            self.assertEqual(len(exemplar_audios), 1)

    def test_mutate_with_auto_exemplar_generation(self) -> None:
        """Verify that mutation with generate_exemplars=True queues engine jobs and parents exemplars to individuals."""
        model_node = StyleGANModel(
            id="model_mutate_auto_ex",
            name="Auto Ex Mutate Model",
            context={},
            checkpoint=Asset(path="mock_model_path", uid="mock_model_uid", extension=".pt"),
            adapter="stylegan2"
        )
        self.graph.add_element(model_node)

        output_dir = self.graph.root / "generate"
        os.makedirs(output_dir, exist_ok=True)
        gene = LoRAGene("layer_mut_auto", torch.randn(2, 2), torch.randn(2, 2), True)
        genome = LoRAGenome([gene])
        gpath = output_dir / "p_mut_auto.safetensors"
        genome.save(str(gpath))

        parent_ind = Individual(
            id="p_mut_auto_ind",
            name="Parent Mut Auto",
            file=Asset(path=str(gpath), uid="p_mut_auto_ind", extension=".safetensors"),
            base_model_id="model_mutate_auto_ex",
            generation=0,
            context={"model_id": "model_mutate_auto_ex", "baseline_elements": []}
        )
        self.graph.add_element(parent_ind)
        self.graph.link(model_node, parent_ind, relation="binds_to")
        self.graph.save()

        self.mock_engine.execute.reset_mock()

        resp = self.client.post("/execute_operation", json={
            "operation": "mutate",
            "parents": [parent_ind.id],
            "offspring_size": 2,
            "generate_exemplars": True,
            "lora_noise": 0.05,
            "mutation_rate": 0.5
        })
        self.assertEqual(resp.status_code, 202)
        job_id = resp.get_json()["job_id"]

        completed_data = None
        for _ in range(50):
            status_resp = self.client.get(f"/job_status/{job_id}")
            if status_resp.status_code == 200:
                s_data = status_resp.get_json()
                if s_data.get("status") == "completed":
                    completed_data = s_data.get("result")
                    break
                elif s_data.get("status") == "failed":
                    self.fail(f"Mutation failed: {s_data.get('error')}")
            time.sleep(0.01)

        self.assertIsNotNone(completed_data)
        self.assertEqual(len(completed_data["individual_ids"]), 2)
        self.assertIn("job_ids", completed_data)
        self.assertEqual(len(completed_data["job_ids"]), 2)

        # Verify engine.execute was called twice (once per child individual)
        self.assertEqual(self.mock_engine.execute.call_count, 2)

        # Poll and complete each exemplar sub-job
        for sub_job_id in completed_data["job_ids"]:
            sub_resp = self.client.get(f"/job_status/{sub_job_id}")
            self.assertEqual(sub_resp.status_code, 200)
            sub_res = sub_resp.get_json()
            self.assertEqual(sub_res["status"], "completed")

        # Reload graph and verify that each child individual has an exemplar audio parented to it
        self.graph.load()
        for child_id in completed_data["individual_ids"]:
            child_node = self.graph.get_element(child_id)
            self.assertIsNotNone(child_node)
            exemplar_audios = [
                node_id for node_id, attrs in self.graph.G.nodes(data=True)
                if attrs.get("parent") == child_id and self.graph.get_element(node_id).type == "audio"
            ]
            self.assertEqual(len(exemplar_audios), 1)

