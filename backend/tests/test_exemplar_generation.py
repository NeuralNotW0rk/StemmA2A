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
from operations.evolution.resolution import find_exemplar_audio, resolve_exemplar_context, collect_exemplar_presets
from diffracture.topology.grating import Grating as DiffractureGrating
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
        self.old_active_jobs = dict(app_module.active_jobs)
        self.old_local_jobs = dict(app_module.local_jobs)
        app_module.active_jobs.clear()
        app_module.local_jobs.clear()
        app_module.param_graph = self.graph
        app_module.engine_provider = self.provider
        app_module.trigger_embedding_update = AsyncMock(return_value=None) if hasattr(app_module.trigger_embedding_update, "__await__") else unittest.mock.MagicMock()

        self.client = app_module.app.test_client()

    def tearDown(self) -> None:
        app_module.param_graph = self.old_graph
        app_module.engine_provider = self.old_provider
        app_module.trigger_embedding_update = self.old_trigger
        app_module.active_jobs = self.old_active_jobs
        app_module.local_jobs = self.old_local_jobs
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
            "mutation_rate": 1.0
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
            "mutation_rate": 1.0
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

    def test_phenotypic_duplicate_safeguard_async_job(self) -> None:
        """Verify that when an offspring generates an identical exemplar to its ancestor, the ancestor's parentage is preserved and the child is flagged as a phenotypic duplicate."""
        model_node = StyleGANModel(
            id="model_dup_test",
            name="Dup Test Model",
            context={},
            checkpoint=Asset(path="mock_model_path", uid="mock_model_uid", extension=".pt"),
            adapter="stylegan2"
        )
        self.graph.add_element(model_node)

        # Create parent individual
        output_dir = self.graph.root / "generate"
        os.makedirs(output_dir, exist_ok=True)
        gene_p = LoRAGene("layer_p", torch.randn(2, 2), torch.randn(2, 2), True)
        genome_p = LoRAGenome([gene_p])
        p_path = output_dir / "p_dup.safetensors"
        genome_p.save(str(p_path))

        parent_ind = Individual(
            id="parent_dup_ind",
            name="Parent Dup Individual",
            file=Asset(path=str(p_path), uid="parent_dup_ind", extension=".safetensors"),
            base_model_id="model_dup_test",
            generation=0,
            context={"model_id": "model_dup_test", "baseline_elements": []}
        )
        self.graph.add_element(parent_ind)
        self.graph.link(model_node, parent_ind, relation="binds_to")

        # Create child individual
        gene_c = LoRAGene("layer_c", torch.randn(2, 2), torch.randn(2, 2), True)
        genome_c = LoRAGenome([gene_c])
        c_path = output_dir / "child_dup.safetensors"
        genome_c.save(str(c_path))

        child_ind = Individual(
            id="child_dup_ind",
            name="Child Dup Individual",
            file=Asset(path=str(c_path), uid="child_dup_ind", extension=".safetensors"),
            base_model_id="model_dup_test",
            generation=1,
            context={"model_id": "model_dup_test", "baseline_elements": []}
        )
        self.graph.add_element(child_ind)
        self.graph.link(parent_ind, child_ind, relation="parent")
        self.graph.save()

        # Deterministic fixed exemplar audio file & id
        shared_audio_id = "audio_shared_identical_uid"
        fpath = os.path.join(self.tmp_dir, "shared_audio.wav")
        with open(fpath, "wb") as f:
            f.write(b"RIFFmockwavdata")

        # Setup mock_engine.get_job_status to return the exact same exemplar artifact for both jobs
        async def _mock_status_same_audio(jid):
            fpath = os.path.join(self.tmp_dir, f"shared_audio_{uuid.uuid4().hex[:6]}.wav")
            with open(fpath, "wb") as f:
                f.write(b"RIFFmockwavdata")
            return {
                "status": "completed",
                "result": {
                    "artifact": {
                        "type": "audio",
                        "id": shared_audio_id,
                        "name": "Shared Exemplar Audio",
                        "file": {"path": fpath, "uid": shared_audio_id, "extension": ".wav"},
                        "context": {}
                    }
                }
            }
        self.mock_engine.get_job_status = AsyncMock(side_effect=_mock_status_same_audio)

        # 1. Generate exemplar for parent
        p_job_id = "job_parent_ex"
        resp1 = self.client.post("/execute_operation", json={
            "job_id": p_job_id,
            "operation": "generate",
            "initiator": parent_ind.to_dict(),
            "params": {}
        })
        self.assertEqual(resp1.status_code, 202)

        p_status_resp = self.client.get(f"/job_status/{p_job_id}")
        self.assertEqual(p_status_resp.status_code, 200)

        # Verify parent owns the exemplar
        self.graph.load()
        self.assertEqual(self.graph.G.nodes[shared_audio_id].get("parent"), parent_ind.id)

        # 2. Generate exemplar for child (which returns the identical shared_audio_id)
        c_job_id = "job_child_ex"
        resp2 = self.client.post("/execute_operation", json={
            "job_id": c_job_id,
            "operation": "generate",
            "initiator": child_ind.to_dict(),
            "params": {}
        })
        self.assertEqual(resp2.status_code, 202)

        c_status_resp = self.client.get(f"/job_status/{c_job_id}")
        self.assertEqual(c_status_resp.status_code, 200)

        # Reload graph and verify that:
        # A) Parent STILL owns the exemplar (it was not stolen)
        self.graph.load()
        self.assertEqual(self.graph.G.nodes[shared_audio_id].get("parent"), parent_ind.id)

        # B) Child node has been flagged as duplicate
        updated_child = self.graph.get_element(child_ind.id)
        self.assertEqual(updated_child.context.get("phenotype_status"), "duplicate")
        self.assertEqual(updated_child.context.get("phenotype_duplicate_of"), parent_ind.id)
        self.assertEqual(updated_child.context.get("shared_exemplar_id"), shared_audio_id)

        # C) Edge exists from child to the shared exemplar with relation 'shares_phenotype'
        edge_data = self.graph.G.get_edge_data(child_ind.id, shared_audio_id)
        self.assertIsNotNone(edge_data)
        self.assertEqual(edge_data.get("relation"), "shares_phenotype")

    def test_wrap_individual_phenotypic_duplicate_safeguard(self) -> None:
        """Verify that wrapping an audio precursor already claimed by another individual flags the new individual as duplicate."""
        model_node = StyleGANModel(
            id="model_wrap_dup",
            name="Wrap Dup Model",
            context={},
            checkpoint=Asset(path="mock_model_path", uid="mock_model_uid", extension=".pt"),
            adapter="stylegan2"
        )
        self.graph.add_element(model_node)

        # Create baseline grating
        diff_grating = DiffractureGrating()
        gpath = os.path.join(self.tmp_dir, "base_grating_wrap.safetensors")
        diff_grating.save(gpath)

        grating_node = Grating(
            id="grating_wrap_dup",
            name="Wrap Baseline Grating",
            context={},
            file=Asset(path=gpath, uid="grating_wrap_dup", extension=".safetensors"),
            base_model_id="model_wrap_dup",
            elements=[]
        )
        self.graph.add_element(grating_node)

        # Create precursor Audio already parented to an existing individual ind_1
        audio_path = os.path.join(self.tmp_dir, "precursor_audio.wav")
        with open(audio_path, "wb") as f:
            f.write(b"RIFFmockwavdata")

        audio_node = Audio(
            id="audio_claimed",
            name="Claimed Audio",
            file=Asset(path=audio_path, uid="audio_claimed", extension=".wav"),
            context={"model_id": "model_wrap_dup"}
        )
        self.graph.add_element(audio_node)
        self.graph.update_element(audio_node.id, {"parent": "original_owner_ind"})
        self.graph.save()

        # Wrap this audio into a new individual ind_2
        resp = self.client.post("/wrap_individual", json={
            "precursor_id": "audio_claimed",
            "model_id": "model_wrap_dup",
            "baseline_grating_id": "grating_wrap_dup",
            "name": "Wrapped Clone Individual"
        })
        self.assertEqual(resp.status_code, 200)
        new_ind_id = resp.get_json()["node_id"]

        self.graph.load()
        # Verify original parentage on audio node was untouched
        self.assertEqual(self.graph.G.nodes["audio_claimed"].get("parent"), "original_owner_ind")

        # Verify new individual was flagged as duplicate
        new_ind = self.graph.get_element(new_ind_id)
        self.assertEqual(new_ind.context.get("phenotype_status"), "duplicate")
        self.assertEqual(new_ind.context.get("phenotype_duplicate_of"), "original_owner_ind")
        self.assertEqual(new_ind.context.get("shared_exemplar_id"), "audio_claimed")

        # Verify edge with shares_phenotype exists
        edge_data = self.graph.G.get_edge_data(new_ind_id, "audio_claimed")
        self.assertIsNotNone(edge_data)
        self.assertEqual(edge_data.get("relation"), "shares_phenotype")

    def test_find_exemplar_audio(self) -> None:
        """Verify find_exemplar_audio discovers direct parented audio, shared phenotype audio, and precursor audio."""
        # 1. Direct parented audio
        ind_1 = Individual(
            id="ind_direct_ex",
            name="Direct Individual",
            file=Asset(path="mock_ind_1", uid="ind_direct_ex", extension=".safetensors"),
            base_model_id="model_test",
            generation=1,
            context={}
        )
        self.graph.add_element(ind_1)

        audio_direct = Audio(
            id="audio_direct_ex",
            name="Direct Audio Exemplar",
            file=Asset(path="mock_audio_1.wav", uid="audio_direct_ex", extension=".wav"),
            context={"prompt": "direct sound"}
        )
        self.graph.add_element(audio_direct)
        self.graph.update_element(audio_direct.id, {"parent": ind_1.id})

        found_audio = find_exemplar_audio(ind_1.id, self.graph)
        self.assertIsNotNone(found_audio)
        self.assertEqual(found_audio.id, audio_direct.id)

        # 2. Shared exemplar via shared_exemplar_id / shares_phenotype edge
        ind_2 = Individual(
            id="ind_shared_ex",
            name="Shared Individual",
            file=Asset(path="mock_ind_2", uid="ind_shared_ex", extension=".safetensors"),
            base_model_id="model_test",
            generation=2,
            context={"shared_exemplar_id": audio_direct.id, "phenotype_status": "duplicate"}
        )
        self.graph.add_element(ind_2)
        self.graph.link(ind_2, audio_direct, relation="shares_phenotype")

        found_shared = find_exemplar_audio(ind_2.id, self.graph)
        self.assertIsNotNone(found_shared)
        self.assertEqual(found_shared.id, audio_direct.id)

        # 3. Precursor audio
        ind_3 = Individual(
            id="ind_prec_ex",
            name="Precursor Individual",
            file=Asset(path="mock_ind_3", uid="ind_prec_ex", extension=".safetensors"),
            base_model_id="model_test",
            generation=0,
            context={"precursor_artifact_id": "audio_precursor_test"}
        )
        audio_precursor = Audio(
            id="audio_precursor_test",
            name="Precursor Audio",
            file=Asset(path="mock_prec.wav", uid="audio_precursor_test", extension=".wav"),
            context={"prompt": "precursor sound"}
        )
        self.graph.add_element(audio_precursor)
        self.graph.add_element(ind_3)
        self.graph.link(audio_precursor, ind_3, relation="precursor")

        found_prec = find_exemplar_audio(ind_3.id, self.graph)
        self.assertIsNotNone(found_prec)
        self.assertEqual(found_prec.id, audio_precursor.id)

    def test_resolve_exemplar_context_hierarchy(self) -> None:
        """Verify resolve_exemplar_context queries parent and deep ancestor exemplar audio contexts."""
        # 1. Gen 0 Parent with exemplar audio
        parent_ind = Individual(
            id="parent_ind_ctx",
            name="Parent Gen 0",
            file=Asset(path="mock_parent.safetensors", uid="parent_ind_ctx", extension=".safetensors"),
            base_model_id="model_ctx_test",
            generation=0,
            context={"baseline_elements": []}
        )
        self.graph.add_element(parent_ind)

        parent_audio = Audio(
            id="parent_audio_ctx",
            name="Parent Audio",
            file=Asset(path="mock_parent.wav", uid="parent_audio_ctx", extension=".wav"),
            context={"prompt": "deep house baseline", "seconds_total": 8.0, "truncation": 0.65}
        )
        self.graph.add_element(parent_audio)
        self.graph.update_element(parent_audio.id, {"parent": parent_ind.id})

        # 2. Gen 1 Child linked via relation='parent' (has NO exemplar yet)
        child_ind = Individual(
            id="child_ind_ctx",
            name="Child Gen 1",
            file=Asset(path="mock_child.safetensors", uid="child_ind_ctx", extension=".safetensors"),
            base_model_id="model_ctx_test",
            generation=1,
            context={"lineage": {"parent_ids": [parent_ind.id]}}
        )
        self.graph.add_element(child_ind)
        self.graph.link(parent_ind, child_ind, relation="parent")

        # Resolving on child should retrieve parent's exemplar audio context
        child_context = resolve_exemplar_context(child_ind.id, self.graph)
        self.assertEqual(child_context.get("prompt"), "deep house baseline")
        self.assertEqual(child_context.get("seconds_total"), 8.0)
        self.assertEqual(child_context.get("truncation"), 0.65)

        # 3. Gen 2 Grandchild linked to Gen 1 Child (which still has no exemplar)
        grandchild_ind = Individual(
            id="grandchild_ind_ctx",
            name="Grandchild Gen 2",
            file=Asset(path="mock_grandchild.safetensors", uid="grandchild_ind_ctx", extension=".safetensors"),
            base_model_id="model_ctx_test",
            generation=2,
            context={"lineage": {"parent_ids": [child_ind.id]}}
        )
        self.graph.add_element(grandchild_ind)
        self.graph.link(child_ind, grandchild_ind, relation="parent")

        # Resolving on grandchild should traverse up to Gen 0 parent's exemplar audio
        grandchild_context = resolve_exemplar_context(grandchild_ind.id, self.graph)
        self.assertEqual(grandchild_context.get("prompt"), "deep house baseline")
        self.assertEqual(grandchild_context.get("seconds_total"), 8.0)
        self.assertEqual(grandchild_context.get("truncation"), 0.65)

    def test_collect_exemplar_presets(self) -> None:
        """Verify lineage exemplars are merged into distinct presets with per-target coverage."""
        form_config = [
            {"name": "prompt", "type": "textarea", "defaultValue": "default prompt"},
            {"name": "seed", "type": "integer", "defaultValue": 0},
            {"name": "init_audio", "type": "node", "filter": {"type": "audio"}},
        ]

        def add_individual(ind_id: str, generation: int, parent: Individual | None = None) -> Individual:
            context = {"precursor_artifact_id": "precursor_audio"}
            if parent:
                context["lineage"] = {"parent_ids": [parent.id]}
            ind = Individual(
                id=ind_id, name=ind_id, generation=generation, base_model_id="model_test", context=context,
                file=Asset(path=f"{ind_id}.safetensors", uid=ind_id, extension=".safetensors"),
            )
            self.graph.add_element(ind)
            if parent:
                self.graph.link(parent, ind, relation="parent")
            return ind

        def add_exemplar(audio_id: str, owner: Individual, context: dict) -> None:
            audio = Audio(id=audio_id, name=audio_id, context=context,
                          file=Asset(path=f"{audio_id}.wav", uid=audio_id, extension=".wav"))
            self.graph.add_element(audio)
            self.graph.update_element(audio_id, {"parent": owner.id})

        root = add_individual("root", 0)
        add_exemplar("precursor_audio", root, {"prompt": "kick", "seed": 1, "individual_ids": ["root"]})
        add_exemplar("root_snare", root, {"prompt": "snare", "init_audio_id": "some_audio"})
        child_a = add_individual("child_a", 1, root)
        child_b = add_individual("child_b", 1, root)
        # Same parameters as the root's kick exemplar, generated on child_a
        add_exemplar("child_a_kick", child_a, {"prompt": "kick", "seed": 1, "individual_ids": ["child_a"]})

        presets = collect_exemplar_presets([child_a.id, child_b.id], self.graph, form_config)

        self.assertEqual(len(presets), 2)
        kick, snare = presets
        self.assertEqual(kick["params"], {"prompt": "kick", "seed": 1, "init_audio_id": None})
        self.assertEqual(kick["covered_ids"], ["child_a"])
        self.assertEqual([(s["audio_id"], s["distance"]) for s in kick["sources"]],
                         [("child_a_kick", 0), ("precursor_audio", 1)])
        # Missing seed falls back to the field default; node inputs are kept by ID
        self.assertEqual(snare["params"], {"prompt": "snare", "seed": 0, "init_audio_id": "some_audio"})
        self.assertEqual(snare["covered_ids"], [])
        # Descendants inherit precursor_artifact_id in context, but the precursor stays the root's exemplar
        self.assertEqual(collect_exemplar_presets([child_b.id], self.graph, form_config)[0]["covered_ids"], [])

    def test_exemplar_generation_inherits_parent_exemplar_context(self) -> None:
        """Verify that dispatching generate for an individual inherits parent exemplar parameters and merges overrides."""
        model_node = StyleGANModel(
            id="model_inherit_test",
            name="Inherit Test Model",
            context={},
            checkpoint=Asset(path="mock_model_path", uid="mock_model_uid", extension=".pt"),
            adapter="stylegan2"
        )
        self.graph.add_element(model_node)

        parent_ind = Individual(
            id="ind_parent_for_dispatch",
            name="Parent Ind For Dispatch",
            file=Asset(path="mock_parent.safetensors", uid="ind_parent_for_dispatch", extension=".safetensors"),
            base_model_id="model_inherit_test",
            generation=0,
            context={"baseline_elements": []}
        )
        self.graph.add_element(parent_ind)
        self.graph.link(model_node, parent_ind, relation="binds_to")

        parent_audio = Audio(
            id="audio_parent_dispatch",
            name="Parent Audio Dispatch",
            file=Asset(path="mock_parent_audio.wav", uid="audio_parent_dispatch", extension=".wav"),
            context={"truncation": 0.72}
        )
        self.graph.add_element(parent_audio)
        self.graph.update_element(parent_audio.id, {"parent": parent_ind.id})

        child_ind = Individual(
            id="ind_child_for_dispatch",
            name="Child Ind For Dispatch",
            file=Asset(path="mock_child.safetensors", uid="ind_child_for_dispatch", extension=".safetensors"),
            base_model_id="model_inherit_test",
            generation=1,
            context={"baseline_elements": []}
        )
        self.graph.add_element(child_ind)
        self.graph.link(model_node, child_ind, relation="binds_to")
        self.graph.link(parent_ind, child_ind, relation="parent")
        self.graph.save()

        # Trigger generate on child WITHOUT supplying truncation (should inherit 0.72 from parent exemplar)
        gen_job_id = f"job_inherit_{uuid.uuid4().hex[:8]}"
        resp = self.client.post("/execute_operation", json={
            "job_id": gen_job_id,
            "operation": "generate",
            "initiator": child_ind.to_dict(),
            "params": {}
        })
        self.assertEqual(resp.status_code, 202)

        # Check that engine was called with inherited truncation=0.72
        self.mock_engine.execute.assert_called()
        call_args, call_kwargs = self.mock_engine.execute.call_args
        self.assertEqual(call_kwargs.get("truncation"), 0.72)




