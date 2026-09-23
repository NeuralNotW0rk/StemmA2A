import unittest
import torch
import tempfile
import shutil
import os
import time
import uuid
from unittest.mock import AsyncMock, MagicMock

from neutral_selection.representation.individual import Individual as NSIndividual
from neutral_selection.representation.population import Population
from neutral_selection.variation.selection import TournamentSelection, TruncationSelection
from neutral_selection.variation.recombination import RandomNPointCrossover

from evolution.lora_genome import (
    LoRAGene,
    PerturbationGene,
    LoRAGenome,
    get_lora_mutation_strategy,
    get_lora_crossover_strategy,
)
from operations.evolution.recombination import (
    recombine_offspring,
    build_selection_strategy,
    build_crossover_strategy,
    RecombinedOffspring,
    ReproducedOffspring,
    breed_offspring,
)


class TestRecombinationEngine(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(42)

    def _create_mock_lora_individual(self, val: float, fitness: float | None = None) -> NSIndividual:
        genes = [
            LoRAGene("layer1", torch.full((2, 2), val), torch.full((2, 2), val), True),
            LoRAGene("layer2", torch.full((2, 2), val * 2.0), torch.full((2, 2), val * 2.0), True),
        ]
        genome = LoRAGenome(genes)
        ind = NSIndividual(genotype=genome)
        ind.fitness = fitness
        return ind

    def test_recombine_offspring_basic(self) -> None:
        """Verify that recombine_offspring produces the expected number of offspring with lineage."""
        p1 = self._create_mock_lora_individual(1.0, fitness=10.0)
        p2 = self._create_mock_lora_individual(2.0, fitness=20.0)
        p3 = self._create_mock_lora_individual(3.0, fitness=30.0)

        parents = [p1, p2, p3]
        parent_ids = ["ind_1", "ind_2", "ind_3"]

        selection_strat = TournamentSelection(tournament_size=2)
        crossover_strat = get_lora_crossover_strategy(num_cut_points=1)
        mutation_strat = get_lora_mutation_strategy(lora_noise=0.05, active_flip_prob=0.0)

        offspring = recombine_offspring(
            parents=parents,
            offspring_count=5,
            selection_strategy=selection_strat,
            crossover_strategy=crossover_strat,
            mutation_strategy=mutation_strat,
            crossover_prob=1.0,
            elitism=0,
            parent_ids=parent_ids,
        )

        self.assertEqual(len(offspring), 5)
        for item in offspring:
            self.assertIsInstance(item, RecombinedOffspring)
            self.assertIsInstance(item, ReproducedOffspring)  # Alias check
            self.assertIsInstance(item.individual.genotype, LoRAGenome)
            self.assertTrue(item.lineage.crossover_applied)
            self.assertTrue(item.lineage.mutated)
            self.assertGreaterEqual(len(item.lineage.parent_ids), 1)

    def test_recombine_offspring_elitism(self) -> None:
        """Verify that elitism preserves the exact top fitness individuals."""
        p1 = self._create_mock_lora_individual(1.0, fitness=10.0)
        p2 = self._create_mock_lora_individual(2.0, fitness=50.0)  # Highest fitness
        p3 = self._create_mock_lora_individual(3.0, fitness=30.0)

        parents = [p1, p2, p3]
        parent_ids = ["ind_1", "ind_2", "ind_3"]

        selection_strat = TournamentSelection(tournament_size=2)
        crossover_strat = get_lora_crossover_strategy(num_cut_points=1)
        mutation_strat = get_lora_mutation_strategy(lora_noise=0.1, active_flip_prob=0.0)

        offspring = recombine_offspring(
            parents=parents,
            offspring_count=4,
            selection_strategy=selection_strat,
            crossover_strategy=crossover_strat,
            mutation_strategy=mutation_strat,
            crossover_prob=1.0,
            elitism=1,
            parent_ids=parent_ids,
        )

        self.assertEqual(len(offspring), 4)

        # First offspring must be the elite individual (p2 with fitness 50.0)
        elite_record = offspring[0]
        self.assertEqual(elite_record.lineage.parent_ids, ["ind_2"])
        self.assertFalse(elite_record.lineage.crossover_applied)
        self.assertFalse(elite_record.lineage.mutated)
        self.assertEqual(elite_record.individual.fitness, 50.0)

        # Weights must be exactly equal to p2
        for elite_gene, p2_gene in zip(elite_record.individual.genotype, p2.genotype):
            self.assertTrue(torch.equal(elite_gene.lora_down, p2_gene.lora_down))
            self.assertTrue(torch.equal(elite_gene.lora_up, p2_gene.lora_up))

    def test_recombine_offspring_no_crossover(self) -> None:
        """Verify breeding with crossover probability 0.0 (mutation only)."""
        p1 = self._create_mock_lora_individual(1.0, fitness=10.0)
        p2 = self._create_mock_lora_individual(2.0, fitness=20.0)

        parents = [p1, p2]
        parent_ids = ["ind_1", "ind_2"]

        selection_strat = TournamentSelection(tournament_size=2)
        crossover_strat = get_lora_crossover_strategy(num_cut_points=1)
        mutation_strat = get_lora_mutation_strategy(lora_noise=0.05, active_flip_prob=0.0)

        offspring = recombine_offspring(
            parents=parents,
            offspring_count=2,
            selection_strategy=selection_strat,
            crossover_strategy=crossover_strat,
            mutation_strategy=mutation_strat,
            crossover_prob=0.0,
            elitism=0,
            parent_ids=parent_ids,
        )

        self.assertEqual(len(offspring), 2)
        for item in offspring:
            self.assertFalse(item.lineage.crossover_applied)
            self.assertEqual(len(item.lineage.parent_ids), 1)

    def test_recombine_offspring_fail_fast_validation(self) -> None:
        """Verify fail-fast error boundaries for invalid arguments."""
        p1 = self._create_mock_lora_individual(1.0)
        selection_strat = TournamentSelection(tournament_size=2)

        # Empty population
        with self.assertRaises(ValueError):
            recombine_offspring([], offspring_count=2, selection_strategy=selection_strat)

        # Non-positive offspring count
        with self.assertRaises(ValueError):
            recombine_offspring([p1], offspring_count=0, selection_strategy=selection_strat)

        # Invalid crossover_prob
        with self.assertRaises(ValueError):
            recombine_offspring([p1], offspring_count=2, selection_strategy=selection_strat, crossover_prob=1.5)

        # Invalid elitism count
        with self.assertRaises(ValueError):
            recombine_offspring([p1], offspring_count=2, selection_strategy=selection_strat, elitism=5)

    def test_strategy_builders(self) -> None:
        """Verify strategy factory functions from dictionary configurations."""
        # Selection
        tourn = build_selection_strategy({"type": "tournament", "tournament_size": 3})
        self.assertIsInstance(tourn, TournamentSelection)
        self.assertEqual(tourn.tournament_size, 3)

        trunc = build_selection_strategy({"type": "truncation", "k": 4})
        self.assertIsInstance(trunc, TruncationSelection)
        self.assertEqual(trunc.top_k, 4)

        # Crossover
        crossover = build_crossover_strategy({"type": "random_n_point", "num_cut_points": 2})
        self.assertIsInstance(crossover, RandomNPointCrossover)
        self.assertEqual(crossover.num_cut_points, 2)

    def test_lora_blend_crossover_strategy(self) -> None:
        """Verify continuous weight blending crossover for LoRAGenome."""
        p1 = self._create_mock_lora_individual(1.0)
        p2 = self._create_mock_lora_individual(3.0)

        crossover_strat = get_lora_crossover_strategy(strategy_type="blend_crossover", blend_factor=0.5)
        child_genome = crossover_strat(p1.genotype, p2.genotype)

        self.assertIsInstance(child_genome, LoRAGenome)
        self.assertEqual(len(child_genome), 2)
        # Expected value: (1.0 * 0.5) + (3.0 * 0.5) = 2.0 for layer1
        for gene in child_genome:
            if gene.address == "layer1":
                self.assertTrue(torch.allclose(gene.lora_down, torch.full((2, 2), 2.0)))
                self.assertTrue(torch.allclose(gene.lora_up, torch.full((2, 2), 2.0)))

    def test_lora_hierarchical_crossover_strategy(self) -> None:
        """Verify default hierarchical crossover strategy for LoRAGenome."""
        p1 = self._create_mock_lora_individual(1.0)
        p2 = self._create_mock_lora_individual(3.0)

        # Default is hierarchical
        crossover_strat = get_lora_crossover_strategy()
        child_genomes = crossover_strat(p1.genotype, p2.genotype)

        self.assertIsInstance(child_genomes, tuple)
        self.assertEqual(len(child_genomes), 2)
        self.assertIsInstance(child_genomes[0], LoRAGenome)
        self.assertIsInstance(child_genomes[1], LoRAGenome)
        self.assertEqual(len(child_genomes[0]), 2)
        self.assertEqual(len(child_genomes[1]), 2)


class TestRecombinationAPI(unittest.TestCase):

    def test_recombine_evolution_api_and_graph_lineage(self) -> None:
        """Test `/recombine_evolution` endpoint, verifying generation increment, group creation, direct parent-child edges, and child context provenance."""
        import app as app_module
        from param_graph.graph import ParameterGraph
        from param_graph.elements.models.stylegan_element import StyleGANModel
        from param_graph.elements.artifacts.individual_element import Individual
        from param_graph.elements.collections.group_element import Group
        from param_graph.elements.artifacts.audio_element import Audio
        from param_graph.elements.base_elements import Asset
        from engine.engine_provider import EngineProvider

        tmp_dir = tempfile.mkdtemp()
        try:
            g = ParameterGraph(tmp_dir)
            provider = EngineProvider(data_root=tmp_dir)
            mock_engine = provider.get_engine()
            mock_engine.execute = AsyncMock(side_effect=lambda *args, **kwargs: kwargs.get("job_id") or str(uuid.uuid4()))

            async def _mock_get_job_status(jid):
                fpath = os.path.join(tmp_dir, f"mock_audio_{uuid.uuid4().hex[:6]}.wav")
                with open(fpath, "wb") as f:
                    f.write(b"RIFFmockwavdata")
                return {
                    "status": "completed",
                    "result": {
                        "artifact": {
                            "type": "audio",
                            "id": f"audio_child_{uuid.uuid4().hex[:6]}",
                            "name": "Generated Audio",
                            "file": {"path": fpath, "uid": f"uid_{uuid.uuid4().hex[:6]}", "extension": ".wav"},
                            "context": {}
                        }
                    }
                }

            mock_engine.get_job_status = AsyncMock(side_effect=_mock_get_job_status)

            old_graph = app_module.param_graph
            old_provider = app_module.engine_provider
            old_trigger = app_module.trigger_embedding_update
            app_module.param_graph = g
            app_module.engine_provider = provider
            app_module.trigger_embedding_update = MagicMock()

            client = app_module.app.test_client()

            # 1. Add model node
            model_node = StyleGANModel(
                id="model_breed_test",
                name="Breed Test Model",
                context={},
                checkpoint=Asset(path="mock_model_path", uid="mock_model_uid", extension=".pt"),
                adapter="stylegan2"
            )
            g.add_element(model_node)

            # 2. Add precursor audio node
            precursor_audio = Audio(
                id="precursor_audio_breed",
                name="Precursor Audio",
                context={},
                file=Asset(path="mock_audio.wav", uid="mock_audio_uid", extension=".wav")
            )
            g.add_element(precursor_audio)

            # 3. Create Gen 0 individuals with saved safetensors genomes
            output_dir = g.root / "generate"
            os.makedirs(output_dir, exist_ok=True)

            parent_ids = []
            for i in range(3):
                gene = LoRAGene(f"layer_{i}", torch.randn(2, 2), torch.randn(2, 2), True)
                genome = LoRAGenome([gene])
                genome_path = output_dir / f"parent_gen0_{i}.safetensors"
                genome.save(str(genome_path))

                ind_node = Individual(
                    id=f"parent_ind_{i}",
                    name=f"Parent Ind {i}",
                    file=Asset(path=str(genome_path), uid=f"parent_ind_{i}", extension=".safetensors"),
                    base_model_id="model_breed_test",
                    generation=0,
                    fitness=float(i * 10),
                    context={
                        "model_id": "model_breed_test",
                        "operation": "generate",
                        "source_audio_id": "precursor_audio_breed",
                        "representation": "lora",
                        "baseline_elements": []
                    }
                )
                g.add_element(ind_node)
                g.link(model_node, ind_node, relation='binds_to')
                parent_ids.append(ind_node.id)

            g.save()

            # 4. Invoke /recombine_evolution with direct parent_ids list and blend_crossover
            payload = {
                "parent_ids": parent_ids,
                "offspring_size": 4,
                "selection_type": "tournament",
                "crossover_type": "blend_crossover",
                "mutation_rate": 0.05
            }

            resp = client.post("/recombine_evolution", json=payload)
            self.assertEqual(resp.status_code, 202)

            res_data = resp.get_json()
            self.assertTrue(res_data["success"])
            self.assertIn("job_id", res_data)

            job_id = res_data["job_id"]
            completed_data = None
            for _ in range(50):
                status_resp = client.get(f"/job_status/{job_id}")
                if status_resp.status_code == 200:
                    status_data = status_resp.get_json()
                    if status_data.get("status") == "completed":
                        completed_data = status_data.get("result")
                        break
                    elif status_data.get("status") == "failed":
                        self.fail(f"Recombination job failed: {status_data.get('error')}\n{status_data.get('traceback')}")
                    time.sleep(0.01)

            self.assertIsNotNone(completed_data, "Recombination job timed out")
            self.assertEqual(completed_data["generation"], 1)
            self.assertEqual(len(completed_data["individual_ids"]), 4)

            # 5. Verify parameter graph structure and relationships
            g_reloaded = ParameterGraph(tmp_dir)
            g_reloaded.load()

            # Gen 1 Group (Visual Container)
            gen1_group = g_reloaded.get_element(completed_data["group_id"])
            self.assertEqual(gen1_group.type, "group")
            self.assertEqual(len(gen1_group.member_ids), 4)

            # Check child individuals
            for child_id in completed_data["individual_ids"]:
                child_node = g_reloaded.get_element(child_id)
                self.assertEqual(child_node.type, "individual")
                self.assertEqual(child_node.generation, 1)

                # Check safetensors file exists on disk
                self.assertTrue(os.path.exists(child_node.file.path))

                # Check genome can be loaded
                loaded_gen = LoRAGenome.load(child_node.file.path)
                self.assertIsInstance(loaded_gen, LoRAGenome)

                # Check direct parent lineage edges exist
                lineage = child_node.context.get("lineage", {})
                for parent_id in lineage.get("parent_ids", []):
                    self.assertTrue(g_reloaded.G.has_edge(parent_id, child_id))
                    self.assertEqual(g_reloaded.G.get_edge_data(parent_id, child_id)["relation"], "parent")

                # Check full recombine_operation provenance in context
                recombine_ctx = child_node.context.get("recombine_operation", {})
                self.assertEqual(recombine_ctx.get("crossover_type"), "blend_crossover")
                self.assertEqual(recombine_ctx.get("selection_type"), "tournament")
                self.assertEqual(recombine_ctx.get("mutation_rate"), 0.05)
                self.assertEqual(recombine_ctx.get("offspring_size"), 4)

            # Check that child exemplar sub-jobs complete cleanly without spurious audio-to-audio source edges
            for sub_job_id in completed_data.get("job_ids", []):
                sub_status_resp = client.get(f"/job_status/{sub_job_id}")
                self.assertEqual(sub_status_resp.status_code, 200)
                sub_data = sub_status_resp.get_json()
                self.assertEqual(sub_data.get("status"), "completed")
                audio_node_id = sub_data.get("node_id")
                self.assertIsNotNone(audio_node_id)
                # Verify no direct source edges from precursor audio to this evolved exemplar audio
                self.assertFalse(app_module.param_graph.G.has_edge(precursor_audio.id, audio_node_id))

            # 6. Test /execute_operation with operation="recombine"
            execute_payload = {
                "operation": "recombine",
                "params": {
                    "parents": parent_ids,
                    "offspring_size": 2,
                    "crossover_type": "layer_crossover",
                    "selection_type": "uniform",
                    "mutation_rate": 0.1
                }
            }
            exec_resp = client.post("/execute_operation", json=execute_payload)
            self.assertEqual(exec_resp.status_code, 202)
            exec_data = exec_resp.get_json()
            self.assertTrue(exec_data["success"])
            exec_job_id = exec_data["job_id"]

            # Wait for execute job to complete
            for _ in range(50):
                status_resp = client.get(f"/job_status/{exec_job_id}")
                if status_resp.status_code == 200:
                    status_data = status_resp.get_json()
                    if status_data.get("status") in ("completed", "failed"):
                        break
                time.sleep(0.01)

            # Restore globals
            app_module.param_graph = old_graph
            app_module.engine_provider = old_provider
            app_module.trigger_embedding_update = old_trigger

        finally:
            shutil.rmtree(tmp_dir)

    def test_recombine_tosses_out_selected_edges(self) -> None:
        """Verify that passing edge IDs, edge dictionaries, or invalid elements in parents is cleanly filtered out."""
        import app as app_module
        from param_graph.graph import ParameterGraph
        from param_graph.elements.models.stylegan_element import StyleGANModel
        from param_graph.elements.artifacts.individual_element import Individual
        from param_graph.elements.artifacts.audio_element import Audio
        from param_graph.elements.base_elements import Asset
        from engine.engine_provider import EngineProvider

        tmp_dir = tempfile.mkdtemp()
        try:
            g = ParameterGraph(tmp_dir)
            provider = EngineProvider(data_root=tmp_dir)
            mock_engine = provider.get_engine()
            mock_engine.execute = AsyncMock(side_effect=lambda *args, **kwargs: kwargs.get("job_id") or str(uuid.uuid4()))

            old_graph = app_module.param_graph
            old_provider = app_module.engine_provider
            old_trigger = app_module.trigger_embedding_update
            app_module.param_graph = g
            app_module.engine_provider = provider
            app_module.trigger_embedding_update = MagicMock()

            client = app_module.app.test_client()

            # Add model
            model_node = StyleGANModel(
                id="model_edge_test",
                name="Edge Test Model",
                context={},
                checkpoint=Asset(path="mock_model_path", uid="mock_model_uid", extension=".pt"),
                adapter="stylegan2"
            )
            g.add_element(model_node)

            # Create 2 parent individuals
            output_dir = g.root / "generate"
            os.makedirs(output_dir, exist_ok=True)
            parent_ids = []
            for i in range(2):
                gene = LoRAGene(f"layer_{i}", torch.randn(2, 2), torch.randn(2, 2), True)
                genome = LoRAGenome([gene])
                genome_path = output_dir / f"parent_edge_{i}.safetensors"
                genome.save(str(genome_path))

                ind = Individual(
                    id=f"ind_edge_{i}",
                    name=f"Parent Edge {i}",
                    file=Asset(path=str(genome_path), uid=f"ind_edge_{i}", extension=".safetensors"),
                    base_model_id="model_edge_test",
                    generation=0,
                    fitness=1.0,
                    context={"model_id": "model_edge_test"}
                )
                g.add_element(ind)
                parent_ids.append(ind.id)
            g.save()

            # Test _extract_individual_parent_ids directly
            mixed_selection = [
                "ind_edge_0",
                "ind_edge_0->ind_edge_1", # edge string
                {"id": "ind_edge_0->ind_edge_1", "source": "ind_edge_0", "target": "ind_edge_1", "type": "individual"}, # edge dict
                {"id": 0, "node": {"id": "ind_edge_1", "type": "individual"}}, # NodeSelectorList item
                "non_existent_id",
            ]
            extracted = app_module._extract_individual_parent_ids(mixed_selection)
            self.assertEqual(extracted, ["ind_edge_0", "ind_edge_1"])

            # Test recombine endpoint with edge IDs mixed in
            resp = client.post("/recombine_evolution", json={
                "parent_ids": mixed_selection,
                "offspring_size": 2,
                "selection_type": "uniform",
                "crossover_type": "two_point"
            })
            self.assertEqual(resp.status_code, 202)
            res_data = resp.get_json()
            self.assertTrue(res_data["success"])

            # Wait for job completion
            job_id = res_data["job_id"]
            completed = False
            for _ in range(50):
                status_resp = client.get(f"/job_status/{job_id}")
                if status_resp.status_code == 200:
                    status_data = status_resp.get_json()
                    if status_data.get("status") == "completed":
                        completed = True
                        break
                    elif status_data.get("status") == "failed":
                        self.fail(f"Recombine failed unexpectedly: {status_data.get('error')}\n{status_data.get('traceback')}")
                elif status_resp.status_code == 500:
                    status_data = status_resp.get_json()
                    self.fail(f"Recombine failed: {status_data.get('error')}\n{status_data.get('traceback')}")
                time.sleep(0.01)
            self.assertTrue(completed, "Recombine job timed out")

            # Restore globals
            app_module.param_graph = old_graph
            app_module.engine_provider = old_provider
            app_module.trigger_embedding_update = old_trigger

        finally:
            shutil.rmtree(tmp_dir)

    def test_recombine_null_fitness_override(self) -> None:
        """Verify that recombination with tournament selection handles parent individuals with null fitness via default_fitness override."""
        import app as app_module
        from param_graph.graph import ParameterGraph
        from param_graph.elements.models.stylegan_element import StyleGANModel
        from param_graph.elements.artifacts.individual_element import Individual
        from param_graph.elements.base_elements import Asset
        from engine.engine_provider import EngineProvider

        tmp_dir = tempfile.mkdtemp()
        try:
            g = ParameterGraph(tmp_dir)
            provider = EngineProvider(data_root=tmp_dir)
            mock_engine = provider.get_engine()
            mock_engine.execute = AsyncMock(side_effect=lambda *args, **kwargs: kwargs.get("job_id") or str(uuid.uuid4()))

            old_graph = app_module.param_graph
            old_provider = app_module.engine_provider
            old_trigger = app_module.trigger_embedding_update
            app_module.param_graph = g
            app_module.engine_provider = provider
            app_module.trigger_embedding_update = MagicMock()

            client = app_module.app.test_client()

            # Register base model
            model_node = StyleGANModel(
                id="model_null_fit_test",
                name="Mock Model",
                adapter="stylegan2",
                checkpoint=Asset(path="mock_model_path", uid="mock_model_uid", extension=".pt"),
                context={}
            )
            g.add_element(model_node)

            # Create 2 parent individuals with fitness=None
            output_dir = g.root / "generate"
            os.makedirs(output_dir, exist_ok=True)
            parents = []
            for i in range(2):
                gene = LoRAGene(f"layer_{i}", torch.randn(2, 2), torch.randn(2, 2), True)
                genome = LoRAGenome([gene])
                genome_path = output_dir / f"parent_null_{i}.safetensors"
                genome.save(str(genome_path))

                ind = Individual(
                    id=f"ind_null_fit_{i}",
                    name=f"Parent Null {i}",
                    file=Asset(path=str(genome_path), uid=f"ind_null_fit_{i}", extension=".safetensors"),
                    base_model_id="model_null_fit_test",
                    generation=0,
                    fitness=None,
                    context={"model_id": "model_null_fit_test"}
                )
                g.add_element(ind)
                parents.append(f"ind_null_fit_{i}")
            g.save()

            # Test recombine with tournament selection (which requires numeric fitness) and custom default_fitness
            resp = client.post("/recombine_evolution", json={
                "parent_ids": parents,
                "offspring_size": 2,
                "selection_type": "tournament",
                "crossover_type": "two_point",
                "default_fitness": 2.5
            })
            self.assertEqual(resp.status_code, 202)
            res_data = resp.get_json()
            self.assertTrue(res_data["success"])

            job_id = res_data["job_id"]
            completed = False
            for _ in range(50):
                status_resp = client.get(f"/job_status/{job_id}")
                if status_resp.status_code == 200:
                    status_data = status_resp.get_json()
                    if status_data.get("status") == "completed":
                        completed = True
                        break
                    elif status_data.get("status") == "failed":
                        self.fail(f"Recombine failed unexpectedly: {status_data.get('error')}\n{status_data.get('traceback')}")
                elif status_resp.status_code == 500:
                    status_data = status_resp.get_json()
                    self.fail(f"Recombine failed: {status_data.get('error')}\n{status_data.get('traceback')}")
                time.sleep(0.01)
            self.assertTrue(completed, "Recombine job timed out")

            # Reload graph and verify child recombine_operation has default_fitness
            g_reloaded = ParameterGraph(tmp_dir)
            g_reloaded.load()
            children = [
                g_reloaded.get_element(nid) for nid in g_reloaded.G.nodes
                if isinstance(g_reloaded.get_element(nid), Individual) and g_reloaded.get_element(nid).generation == 1
            ]
            self.assertEqual(len(children), 2)
            for child in children:
                recomb_ctx = child.context.get("recombine_operation", {})
                self.assertEqual(recomb_ctx.get("default_fitness"), 2.5)

            # Also test /execute_operation with operation='recombine' without specifying default_fitness (should default to 0.0)
            exec_resp = client.post("/execute_operation", json={
                "operation": "recombine",
                "params": {
                    "parents": parents,
                    "offspring_size": 2,
                    "selection_type": "tournament",
                    "crossover_type": "two_point"
                }
            })
            self.assertEqual(exec_resp.status_code, 202)
            exec_data = exec_resp.get_json()
            self.assertTrue(exec_data["success"])

            job_id = exec_data["job_id"]
            completed = False
            for _ in range(50):
                status_resp = client.get(f"/job_status/{job_id}")
                if status_resp.status_code == 200:
                    status_data = status_resp.get_json()
                    if status_data.get("status") == "completed":
                        completed = True
                        break
                    elif status_data.get("status") == "failed":
                        self.fail(f"Recombine via execute_operation failed: {status_data.get('error')}\n{status_data.get('traceback')}")
                time.sleep(0.01)
            self.assertTrue(completed, "Recombine via execute_operation timed out")

            # Restore globals
            app_module.param_graph = old_graph
            app_module.engine_provider = old_provider
            app_module.trigger_embedding_update = old_trigger

        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
