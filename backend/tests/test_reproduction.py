import unittest
import torch
import tempfile
import shutil
import os
import time
from unittest.mock import AsyncMock

from neutral_selection.representation.individual import Individual as NSIndividual
from neutral_selection.representation.population import Population
from neutral_selection.variation.selection import TournamentSelection, TruncationSelection
from neutral_selection.variation.recombination import RandomNPointCrossover

from evolution.lora.lora_genome import (
    PerturbationGene,
    LoRAGenome,
    get_lora_mutation_strategy,
    get_lora_crossover_strategy,
)
from evolution.reproduction import (
    breed_offspring,
    build_selection_strategy,
    build_crossover_strategy,
    ReproducedOffspring,
)
from evolution.registry import (
    get_genome_class,
    get_expression_function,
    register_representation,
)


class TestReproductionEngine(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(42)

    def _create_mock_lora_individual(self, val: float, fitness: float | None = None) -> NSIndividual:
        genes = [
            PerturbationGene("layer1", torch.full((2, 2), val), torch.full((2, 2), val), True),
            PerturbationGene("layer2", torch.full((2, 2), val * 2.0), torch.full((2, 2), val * 2.0), True),
        ]
        genome = LoRAGenome(genes)
        ind = NSIndividual(genotype=genome)
        ind.fitness = fitness
        return ind

    def test_breed_offspring_basic(self) -> None:
        """Verify that breed_offspring produces the expected number of offspring with lineage."""
        p1 = self._create_mock_lora_individual(1.0, fitness=10.0)
        p2 = self._create_mock_lora_individual(2.0, fitness=20.0)
        p3 = self._create_mock_lora_individual(3.0, fitness=30.0)

        parents = [p1, p2, p3]
        parent_ids = ["ind_1", "ind_2", "ind_3"]

        selection_strat = TournamentSelection(tournament_size=2)
        crossover_strat = get_lora_crossover_strategy(num_cut_points=1)
        mutation_strat = get_lora_mutation_strategy(lora_noise=0.05, active_flip_prob=0.0)

        offspring = breed_offspring(
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
            self.assertIsInstance(item, ReproducedOffspring)
            self.assertIsInstance(item.individual.genotype, LoRAGenome)
            self.assertTrue(item.lineage.crossover_applied)
            self.assertTrue(item.lineage.mutated)
            self.assertGreaterEqual(len(item.lineage.parent_ids), 1)

    def test_breed_offspring_elitism(self) -> None:
        """Verify that elitism preserves the exact top fitness individuals."""
        p1 = self._create_mock_lora_individual(1.0, fitness=10.0)
        p2 = self._create_mock_lora_individual(2.0, fitness=50.0)  # Highest fitness
        p3 = self._create_mock_lora_individual(3.0, fitness=30.0)

        parents = [p1, p2, p3]
        parent_ids = ["ind_1", "ind_2", "ind_3"]

        selection_strat = TournamentSelection(tournament_size=2)
        crossover_strat = get_lora_crossover_strategy(num_cut_points=1)
        mutation_strat = get_lora_mutation_strategy(lora_noise=0.1, active_flip_prob=0.0)

        offspring = breed_offspring(
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

    def test_breed_offspring_no_crossover(self) -> None:
        """Verify breeding with crossover probability 0.0 (mutation only)."""
        p1 = self._create_mock_lora_individual(1.0, fitness=10.0)
        p2 = self._create_mock_lora_individual(2.0, fitness=20.0)

        parents = [p1, p2]
        parent_ids = ["ind_1", "ind_2"]

        selection_strat = TournamentSelection(tournament_size=2)
        crossover_strat = get_lora_crossover_strategy(num_cut_points=1)
        mutation_strat = get_lora_mutation_strategy(lora_noise=0.05, active_flip_prob=0.0)

        offspring = breed_offspring(
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

    def test_breed_offspring_fail_fast_validation(self) -> None:
        """Verify fail-fast error boundaries for invalid arguments."""
        p1 = self._create_mock_lora_individual(1.0)
        selection_strat = TournamentSelection(tournament_size=2)

        # Empty population
        with self.assertRaises(ValueError):
            breed_offspring([], offspring_count=2, selection_strategy=selection_strat)

        # Non-positive offspring count
        with self.assertRaises(ValueError):
            breed_offspring([p1], offspring_count=0, selection_strategy=selection_strat)

        # Invalid crossover_prob
        with self.assertRaises(ValueError):
            breed_offspring([p1], offspring_count=2, selection_strategy=selection_strat, crossover_prob=1.5)

        # Invalid elitism count
        with self.assertRaises(ValueError):
            breed_offspring([p1], offspring_count=2, selection_strategy=selection_strat, elitism=5)

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

    def test_registry_integration(self) -> None:
        """Verify representation registry retrieval."""
        genome_cls = get_genome_class("lora")
        self.assertEqual(genome_cls, LoRAGenome)

        expr_fn = get_expression_function("lora")
        self.assertTrue(callable(expr_fn))

        with self.assertRaises(ValueError):
            get_genome_class("nonexistent_representation")


class TestReproductionAPI(unittest.TestCase):

    def test_reproduce_evolution_api_and_graph_lineage(self) -> None:
        """Test `/reproduce_evolution` endpoint, verifying generation increment, bundle/group creation, and parent-child edges."""
        import app as app_module
        from param_graph.graph import ParameterGraph
        from param_graph.elements.models.stylegan_element import StyleGANModel
        from param_graph.elements.artifacts.individual_element import Individual
        from param_graph.elements.artifacts.bundle_element import Bundle
        from param_graph.elements.collections.group_element import Group
        from param_graph.elements.artifacts.audio_element import Audio
        from param_graph.elements.base_elements import Asset
        from engine.engine_provider import EngineProvider

        tmp_dir = tempfile.mkdtemp()
        try:
            g = ParameterGraph(tmp_dir)
            provider = EngineProvider(data_root=tmp_dir)
            mock_engine = provider.get_engine()
            mock_engine.execute = AsyncMock(return_value="mock-breed-job")

            old_graph = app_module.param_graph
            old_provider = app_module.engine_provider
            app_module.param_graph = g
            app_module.engine_provider = provider

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
                gene = PerturbationGene(f"layer_{i}", torch.randn(2, 2), torch.randn(2, 2), True)
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

            # Gen 0 Bundle
            gen0_bundle = Bundle(
                id="bundle_gen0",
                name="Generation 0 Bundle",
                member_ids=parent_ids,
                member_type='individual',
                context={"generation": 0}
            )
            g.add_element(gen0_bundle)
            g.link(precursor_audio, gen0_bundle, relation='precursor')
            for pid in parent_ids:
                g.link(gen0_bundle, g.get_element(pid), relation='member')

            g.save()

            # 4. Invoke /reproduce_evolution
            payload = {
                "parent_bundle_id": "bundle_gen0",
                "offspring_size": 4,
                "selection": {
                    "type": "tournament",
                    "tournament_size": 2
                },
                "crossover": {
                    "type": "random_n_point",
                    "num_cut_points": 1,
                    "prob": 0.9
                },
                "mutation": {
                    "lora_noise": 0.05,
                    "active_flip_prob": 0.05,
                    "mutation_rate": 1.0
                },
                "elitism": 1
            }

            resp = client.post("/reproduce_evolution", json=payload)
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
                        self.fail(f"Reproduction job failed: {status_data.get('error')}\n{status_data.get('traceback')}")
                    time.sleep(0.1)

            self.assertIsNotNone(completed_data, "Reproduction job timed out")
            self.assertEqual(completed_data["generation"], 1)
            self.assertEqual(len(completed_data["individual_ids"]), 4)

            # 5. Verify parameter graph structure and relationships
            g_reloaded = ParameterGraph(tmp_dir)
            g_reloaded.load()

            # Gen 1 Bundle
            gen1_bundle = g_reloaded.get_element(completed_data["bundle_id"])
            self.assertEqual(gen1_bundle.type, "bundle")
            self.assertEqual(len(gen1_bundle.member_ids), 4)

            # Gen 1 Group
            gen1_group = g_reloaded.get_element(completed_data["group_id"])
            self.assertEqual(gen1_group.type, "group")
            self.assertEqual(len(gen1_group.member_ids), 4)

            # Bundle -> Bundle edge (next_generation)
            self.assertTrue(g_reloaded.G.has_edge("bundle_gen0", gen1_bundle.id))
            edge_data = g_reloaded.G.get_edge_data("bundle_gen0", gen1_bundle.id)
            self.assertEqual(edge_data["relation"], "next_generation")

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

                # Check parent lineage edges exist
                lineage = child_node.context.get("lineage", {})
                for parent_id in lineage.get("parent_ids", []):
                    self.assertTrue(g_reloaded.G.has_edge(parent_id, child_id))
                    self.assertEqual(g_reloaded.G.get_edge_data(parent_id, child_id)["relation"], "parent")

                # Check bundle membership
                self.assertTrue(g_reloaded.G.has_edge(gen1_bundle.id, child_id))
                self.assertEqual(g_reloaded.G.get_edge_data(gen1_bundle.id, child_id)["relation"], "member")

            # Restore globals
            app_module.param_graph = old_graph
            app_module.engine_provider = old_provider

        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
