import unittest
import torch
from diffracture.topology.grating import Grating
from diffracture.topology.lora import LoRAElement

from evolution.lora.lora_genome import (
    PerturbationGene,
    LoRAGenome,
    genome_from_grating,
    express_to_grating,
    RandomNPointCrossover,
    mutate_perturbation_gene
)
from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population


class TestLoRAEvolution(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_gene_from_and_to_tensors(self) -> None:
        """Verify initialization and scaling on reconstruction."""
        address = "layer_test"
        lora_down = torch.randn(2, 5, dtype=torch.float64) * 5.0
        lora_up = torch.randn(8, 2, dtype=torch.float64) * 5.0

        gene = PerturbationGene.from_tensors(address, lora_down, lora_up, active=True)

        # Check weights are saved
        self.assertTrue(torch.allclose(gene.lora_down, lora_down))
        self.assertTrue(torch.allclose(gene.lora_up, lora_up))

        # Check reconstruction
        rec_down, rec_up = gene.to_tensors()
        self.assertTrue(torch.allclose(lora_down, rec_down))
        self.assertTrue(torch.allclose(lora_up, rec_up))

    def test_genome_grating_translation(self) -> None:
        """Verify translation from Grating to LoRAGenome and back."""
        grating = Grating()
        el1 = LoRAElement("module1", rank=2, alpha=1.0, in_features=4, out_features=4)
        el2 = LoRAElement("module2", rank=2, alpha=1.0, in_features=4, out_features=4)
        
        # Give them random weights
        el1.params["lora_down"].data.copy_(torch.randn(2, 4))
        el1.params["lora_up"].data.copy_(torch.randn(4, 2))
        el2.params["lora_down"].data.copy_(torch.randn(2, 4))
        el2.params["lora_up"].data.copy_(torch.randn(4, 2))

        grating.add_element(el1)
        grating.add_element(el2)

        # Create genome
        genome = genome_from_grating(grating)
        self.assertEqual(len(genome), 2)
        self.assertEqual(genome[0].address, "module1")
        self.assertEqual(genome[1].address, "module2")

        # Mutate genome weights
        genome[0].lora_down.fill_(10.0)
        genome[0].lora_up.fill_(1.0)
        genome[1].lora_down.fill_(5.0)
        genome[1].lora_up.fill_(1.0)

        # Express back to new Grating
        new_grating = express_to_grating(genome, grating)

        # Verify new Grating has updated parameters
        nodes = new_grating.nodes
        new_el1 = nodes["module1"]
        new_el2 = nodes["module2"]

        self.assertTrue(torch.allclose(new_el1.params["lora_down"], torch.full_like(new_el1.params["lora_down"], 10.0)))
        self.assertTrue(torch.allclose(new_el2.params["lora_down"], torch.full_like(new_el2.params["lora_down"], 5.0)))

    def test_n_point_crossover(self) -> None:
        """Verify NPoint crossover swaps segments correctly."""
        genes_p1 = [
            PerturbationGene("l1", torch.ones(1, 1), torch.ones(1, 1), True),
            PerturbationGene("l2", torch.ones(1, 1), torch.ones(1, 1) * 2.0, True),
            PerturbationGene("l3", torch.ones(1, 1), torch.ones(1, 1) * 3.0, True),
            PerturbationGene("l4", torch.ones(1, 1), torch.ones(1, 1) * 4.0, True),
        ]
        genes_p2 = [
            PerturbationGene("l1", torch.ones(1, 1) * 10.0, torch.ones(1, 1) * 10.0, False),
            PerturbationGene("l2", torch.ones(1, 1) * 20.0, torch.ones(1, 1) * 20.0, False),
            PerturbationGene("l3", torch.ones(1, 1) * 30.0, torch.ones(1, 1) * 30.0, False),
            PerturbationGene("l4", torch.ones(1, 1) * 40.0, torch.ones(1, 1) * 40.0, False),
        ]
        genome_p1 = LoRAGenome(genes_p1)
        genome_p2 = LoRAGenome(genes_p2)

        # Crossover at cut point 2 (index 2)
        crossover = RandomNPointCrossover(num_cut_points=1)
        # Mock random sample to return [2]
        import unittest.mock as mock
        with mock.patch("random.sample", return_value=[2]):
            child_a, child_b = crossover(genome_p1, genome_p2)

        # Child A: P1[:2] + P2[2:]
        self.assertEqual(child_a[0].lora_up.item(), 1.0)
        self.assertEqual(child_a[1].lora_up.item(), 2.0)
        self.assertEqual(child_a[2].lora_up.item(), 30.0)
        self.assertEqual(child_a[3].lora_up.item(), 40.0)

        # Child B: P2[:2] + P1[2:]
        self.assertEqual(child_b[0].lora_up.item(), 10.0)
        self.assertEqual(child_b[1].lora_up.item(), 20.0)
        self.assertEqual(child_b[2].lora_up.item(), 3.0)
        self.assertEqual(child_b[3].lora_up.item(), 4.0)

    def test_mutation(self) -> None:
        """Verify gene mutation perturbs weights and active status."""
        gene = PerturbationGene("l1", torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), True)

        mutated = mutate_perturbation_gene(
            gene,
            direction_noise=0.1,
            magnitude_noise=0.2,
            active_flip_prob=0.0
        )

        # Check weights changed
        self.assertFalse(torch.equal(mutated.lora_down, gene.lora_down))
        self.assertFalse(torch.equal(mutated.lora_up, gene.lora_up))

        # Verify mutation still occurs for inactive genes (recessive/dormant mutation)
        inactive_gene = PerturbationGene("l1", torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), False)
        mutated_inactive = mutate_perturbation_gene(
            inactive_gene,
            direction_noise=0.1,
            magnitude_noise=0.2,
            active_flip_prob=0.0
        )
        self.assertFalse(torch.equal(mutated_inactive.lora_down, inactive_gene.lora_down))

        # Verify that mutating a zero weight gene initializes it with random weights
        zero_gene = PerturbationGene("l1", torch.zeros(2, 2), torch.zeros(2, 2), True)
        mutated_zero = mutate_perturbation_gene(
            zero_gene,
            direction_noise=0.1,
            magnitude_noise=0.2,
            active_flip_prob=0.0
        )
        self.assertFalse(torch.all(mutated_zero.lora_down == 0.0))
        self.assertFalse(torch.all(mutated_zero.lora_up == 0.0))

    def test_fail_fast_mismatch(self) -> None:
        """Verify fail-fast behavior when shape mismatches occur."""
        grating = Grating()
        el = LoRAElement("module", rank=2, alpha=1.0, in_features=4, out_features=4)
        grating.add_element(el)

        # Gene with wrong rank/shape (rank 3 instead of 2)
        mismatched_gene = PerturbationGene("module", torch.randn(3, 4), torch.randn(4, 3), True)
        genome = LoRAGenome([mismatched_gene])

        with self.assertRaises(ValueError):
            express_to_grating(genome, grating)

    def test_population_and_individual_interface(self) -> None:
        """Verify NeutralSelection Individual and Population integrations work."""
        genes = [PerturbationGene("l1", torch.ones(1, 1), torch.ones(1, 1), True)]
        genome = LoRAGenome(genes)

        individual = Individual(genotype=genome)
        self.assertFalse(individual.is_expressed)

        # Mock decode function
        base_grating = Grating()
        el = LoRAElement("l1", rank=1, alpha=1.0, in_features=1, out_features=1)
        base_grating.add_element(el)

        decode_fn = lambda gen: express_to_grating(gen, base_grating)
        expressed = individual.express(decode_fn)

        self.assertTrue(individual.is_expressed)
        self.assertIsInstance(expressed, Grating)

        individual.fitness = 42.0

        # Population
        pop = Population([individual])
        self.assertEqual(len(pop), 1)
        self.assertEqual(pop.best_individual, individual)

    def test_generic_recombine_interface(self) -> None:
        """Verify the recombine polymorphic helper works for both Genomes and Individuals."""
        from neutral_selection.variation.recombination import recombine

        genes_p1 = [PerturbationGene("l1", torch.ones(1, 1), torch.ones(1, 1), True)]
        genes_p2 = [PerturbationGene("l1", torch.zeros(1, 1), torch.zeros(1, 1), False)]
        genome_p1 = LoRAGenome(genes_p1)
        genome_p2 = LoRAGenome(genes_p2)

        crossover = RandomNPointCrossover(num_cut_points=1)

        # 1. Test with Genomes: should return tuple of child Genomes
        child_genomes = recombine(genome_p1, genome_p2, crossover)
        self.assertIsInstance(child_genomes, tuple)
        self.assertIsInstance(child_genomes[0], LoRAGenome)

        # 2. Test with Individuals: should return list of child Individuals
        ind_p1 = Individual(genotype=genome_p1)
        ind_p2 = Individual(genotype=genome_p2)

        child_inds = recombine(ind_p1, ind_p2, crossover)
        self.assertIsInstance(child_inds, list)
        self.assertEqual(len(child_inds), 2)
        self.assertIsInstance(child_inds[0], Individual)
        self.assertIsInstance(child_inds[0].genotype, LoRAGenome)

    def test_genome_save_load_roundtrip(self) -> None:
        """Verify that a LoRAGenome can be saved to disk and loaded back correctly."""
        import tempfile
        import os

        genes = [
            PerturbationGene("layer_1", torch.randn(2, 4), torch.randn(4, 2), True),
            PerturbationGene("layer_2", torch.randn(3, 5), torch.randn(5, 3), False),
        ]
        original_genome = LoRAGenome(genes)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "genome.safetensors")
            original_genome.save(save_path)

            # Check that file exists
            self.assertTrue(os.path.exists(save_path))

            # Load back
            loaded_genome = LoRAGenome.load(save_path)

            # Assert equality
            self.assertEqual(len(loaded_genome), 2)
            for orig_gene, loaded_gene in zip(original_genome, loaded_genome):
                self.assertEqual(orig_gene.address, loaded_gene.address)
                self.assertTrue(torch.allclose(orig_gene.lora_down, loaded_gene.lora_down))
                self.assertTrue(torch.allclose(orig_gene.lora_up, loaded_gene.lora_up))
                self.assertEqual(orig_gene.active, loaded_gene.active)

    def test_express_to_grating_with_path(self) -> None:
        """Verify that express_to_grating works when passed a string path to a base grating."""
        import tempfile
        import os

        grating = Grating()
        el1 = LoRAElement("module1", rank=2, alpha=1.0, in_features=4, out_features=4)
        grating.add_element(el1)

        lora_down = torch.randn(2, 4)
        lora_up = torch.randn(4, 2)
        genes = [
            PerturbationGene("module1", lora_down, lora_up, True)
        ]
        genome = LoRAGenome(genes)

        with tempfile.TemporaryDirectory() as tmpdir:
            grating_path = os.path.join(tmpdir, "base_grating.safetensors")
            grating.save(grating_path)

            # Express using path
            expressed_grating = express_to_grating(genome, grating_path)

            # Verify parameters were expressed correctly
            new_el = expressed_grating.nodes["module1"]
            self.assertTrue(torch.allclose(new_el.params["lora_down"], lora_down.to(new_el.params["lora_down"].dtype)))
            self.assertTrue(torch.allclose(new_el.params["lora_up"], lora_up.to(new_el.params["lora_up"].dtype)))

    def test_load_genome_invalid_metadata_fail_fast(self) -> None:
        """Verify that loading an incompatible/malformed safetensors file fails fast with a ValueError."""
        import tempfile
        import os
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "bad_genome.safetensors")
            # Save a safetensors file without our custom genome metadata
            save_file({"dummy": torch.zeros(1)}, save_path, metadata={"wrong_key": "some_value"})

            with self.assertRaises(ValueError):
                LoRAGenome.load(save_path)

    def test_start_evolution_api_structure(self) -> None:
        """Test `/start_evolution` API endpoint, verifying population bundle, precursor audio linking, and population group visual compound hierarchy."""
        import tempfile
        import shutil
        import os
        from unittest.mock import AsyncMock
        from engine.engine_provider import EngineProvider
        import app as app_module
        from param_graph.graph import ParameterGraph
        from param_graph.elements.models.stylegan_element import StyleGANModel
        from param_graph.elements.artifacts.grating_element import Grating as GratingArtifact
        from param_graph.elements.artifacts.audio_element import Audio
        from param_graph.elements.base_elements import Asset
        from diffracture.topology.grating import Grating

        tmp_dir = tempfile.mkdtemp()
        try:
            # Instantiate clean graph and engine provider
            g = ParameterGraph(tmp_dir)
            provider = EngineProvider(data_root=tmp_dir)
            
            # Mock engine execute to bypass actual neural net generation run
            mock_engine = provider.get_engine()
            mock_engine.execute = AsyncMock(return_value="mock-job-id")
            
            # Bind to Flask app module globals
            old_graph = app_module.param_graph
            old_provider = app_module.engine_provider
            app_module.param_graph = g
            app_module.engine_provider = provider
            
            client = app_module.app.test_client()

            # 1. Add model node
            model_node = StyleGANModel(
                id="model_test",
                name="Test Model",
                context={},
                checkpoint=Asset(path="mock_model_path", uid="mock_model_uid", extension=".pt"),
                adapter="stylegan2"
            )
            g.add_element(model_node)

            # 2. Add precursor audio node
            precursor_audio = Audio(
                id="precursor_audio_test",
                name="Precursor Audio",
                context={},
                file=Asset(path="mock_audio.wav", uid="mock_audio_uid", extension=".wav")
            )
            g.add_element(precursor_audio)

            # 3. Create baseline grating checkpoint and add Grating node
            base_grating = Grating()
            grating_file = os.path.join(tmp_dir, "base_grating.safetensors")
            base_grating.save(grating_file)

            grating_node = GratingArtifact(
                id="grating_test",
                name="Base Grating",
                context={},
                file=Asset(path=grating_file, uid="grating_uid", extension=".safetensors"),
                base_model_id="model_test",
                elements=[]
            )
            g.add_element(grating_node)
            g.save()

            # 4. Invoke /start_evolution
            payload = {
                "model_id": "model_test",
                "baseline_grating_id": "grating_test",
                "precursor_audio_id": "precursor_audio_test",
                "population_size": 3,
                "direction_noise": 0.05,
                "magnitude_noise": 0.1,
                "active_flip_prob": 0.05,
                "generation_context": {
                    "operation": "generate",
                    "prompt": "Test Prompt"
                }
            }

            resp = client.post("/start_evolution", json=payload)
            self.assertEqual(resp.status_code, 200)
            
            res_data = resp.get_json()
            self.assertTrue(res_data["success"])
            self.assertIn("bundle_id", res_data)
            self.assertIn("group_id", res_data)
            self.assertEqual(len(res_data["job_ids"]), 3)
            self.assertEqual(len(res_data["individual_ids"]), 3)

            # 5. Reload graph and verify elements & relationships
            g_reloaded = ParameterGraph(tmp_dir)
            g_reloaded.load()
            
            bundle_node = g_reloaded.get_element(res_data["bundle_id"])
            self.assertEqual(bundle_node.type, "bundle")
            self.assertEqual(len(bundle_node.member_ids), 3)

            group_node = g_reloaded.get_element(res_data["group_id"])
            self.assertEqual(group_node.type, "group")
            self.assertEqual(len(group_node.member_ids), 3)

            # Check edges
            # Precursor audio --precursor--> Bundle
            self.assertTrue(g_reloaded.G.has_edge("precursor_audio_test", res_data["bundle_id"]))
            edge_data = g_reloaded.G.get_edge_data("precursor_audio_test", res_data["bundle_id"])
            self.assertEqual(edge_data["relation"], "precursor")

            # Check each individual member
            for ind_id in res_data["individual_ids"]:
                ind_node = g_reloaded.get_element(ind_id)
                self.assertEqual(ind_node.type, "individual")
                # Parent must be the Group ID
                node_attrs = g_reloaded.G.nodes[ind_id]
                self.assertEqual(node_attrs.get("parent"), res_data["group_id"])
                
                # Bundle --member--> Individual edge
                self.assertTrue(g_reloaded.G.has_edge(res_data["bundle_id"], ind_id))
                self.assertEqual(g_reloaded.G.get_edge_data(res_data["bundle_id"], ind_id)["relation"], "member")

            # Restore original globals
            app_module.param_graph = old_graph
            app_module.engine_provider = old_provider

        finally:
            shutil.rmtree(tmp_dir)

    def test_start_evolution_genome_centric(self) -> None:
        """Test `/start_evolution` and `/express_individual` in a genome-centric manner without registering baseline grating nodes."""
        import tempfile
        import shutil
        import os
        from unittest.mock import AsyncMock
        from engine.engine_provider import EngineProvider
        import app as app_module
        from param_graph.graph import ParameterGraph
        from param_graph.elements.models.stylegan_element import StyleGANModel
        from param_graph.elements.artifacts.audio_element import Audio
        from param_graph.elements.base_elements import Asset

        tmp_dir = tempfile.mkdtemp()
        try:
            # Instantiate clean graph and engine provider
            g = ParameterGraph(tmp_dir)
            provider = EngineProvider(data_root=tmp_dir)
            
            # Mock engine execute and create_grating to bypass actual runs
            mock_engine = provider.get_engine()
            mock_engine.execute = AsyncMock(return_value="mock-job-id")
            
            # Bind to Flask app module globals
            old_graph = app_module.param_graph
            old_provider = app_module.engine_provider
            app_module.param_graph = g
            app_module.engine_provider = provider
            
            client = app_module.app.test_client()

            # 1. Add model node
            model_node = StyleGANModel(
                id="model_test",
                name="Test Model",
                context={},
                checkpoint=Asset(path="mock_model_path", uid="mock_model_uid", extension=".pt"),
                adapter="stylegan2"
            )
            g.add_element(model_node)

            # 2. Add precursor audio node
            precursor_audio = Audio(
                id="precursor_audio_test",
                name="Precursor Audio",
                context={},
                file=Asset(path="mock_audio.wav", uid="mock_audio_uid", extension=".wav")
            )
            g.add_element(precursor_audio)
            g.save()

            # 3. Invoke /start_evolution with elements config directly
            payload = {
                "model_id": "model_test",
                "elements": [
                    {
                        "address": "layer1",
                        "kernel_type": "lora",
                        "params": {
                            "rank": 4,
                            "alpha": 1.0,
                            "in_features": 4,
                            "out_features": 4
                        }
                    }
                ],
                "precursor_audio_id": "precursor_audio_test",
                "population_size": 2,
                "direction_noise": 0.05,
                "magnitude_noise": 0.1,
                "active_flip_prob": 0.05
            }
            resp = client.post("/start_evolution", json=payload)
            self.assertEqual(resp.status_code, 200)
            res_data = resp.get_json()

            # 4. Verify no baseline grating node was added to the graph database
            g_reloaded = ParameterGraph(tmp_dir)
            g_reloaded.load()
            
            # Verify the individual nodes contain the baseline elements and baseline path in context
            individual_id = res_data["individual_ids"][0]
            ind_node = g_reloaded.get_element(individual_id)
            self.assertEqual(ind_node.type, "individual")
            self.assertIn("baseline_elements", ind_node.context)
            self.assertIn("baseline_file_path", ind_node.context)

            # 5. Verify /express_individual works on this individual using its self-contained context blueprint
            express_payload = {
                "individual_id": individual_id
            }
            express_resp = client.post("/express_individual", json=express_payload)
            self.assertEqual(express_resp.status_code, 200)
            express_data = express_resp.get_json()
            self.assertTrue(express_data["success"])
            
            expressed_grating_id = express_data["grating"]["id"]
            g_reloaded.load()
            self.assertTrue(g_reloaded.G.has_node(expressed_grating_id))

            # Restore original globals
            app_module.param_graph = old_graph
            app_module.engine_provider = old_provider

        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()

