import dataclasses
import unittest
import random
import torch
from unittest.mock import AsyncMock, MagicMock
from diffracture.topology.grating import Grating
from diffracture.topology.lora import LoRAElement

from evolution.genome import (
    LoRAGene,
    PerturbationGene,
    LoRAGenome,
    express_to_grating,
)
from evolution.lora_genome import get_lora_mutation_strategy, lora_gene_scales
from neutral_selection.variation.mutation import mutate
from neutral_selection.variation.recombination import RandomNPointCrossover, recombine
from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population


class TestLoRAEvolution(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_gene_from_and_to_tensors(self) -> None:
        """Verify LoRAGene clones genotype tensors on creation and returns them unchanged."""
        self.assertIs(PerturbationGene, LoRAGene)

        address = "layer_test"
        lora_down = torch.randn(2, 5, dtype=torch.float64) * 5.0
        lora_up = torch.randn(8, 2, dtype=torch.float64) * 5.0

        gene = LoRAGene.from_tensors(address, lora_down, lora_up, active=True)
        self.assertIsNot(gene.lora_down, lora_down)

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

        # Create genome: weights are encoded as genotype values against this (baseline) grating
        genome = LoRAGenome.from_grating(grating)
        self.assertEqual(len(genome), 2)
        self.assertEqual(genome[0].address, "module1")
        self.assertEqual(genome[1].address, "module2")
        scales1 = lora_gene_scales(el1.params["lora_down"], el1.params["lora_up"])
        scales2 = lora_gene_scales(el2.params["lora_down"], el2.params["lora_up"])
        self.assertTrue(torch.allclose(genome[0].lora_down * scales1[0], el1.params["lora_down"]))

        # An unchanged genome expresses back to the baseline's own weights
        roundtrip = express_to_grating(genome, grating).nodes["module1"]
        self.assertTrue(torch.allclose(roundtrip.params["lora_down"], el1.params["lora_down"]))
        self.assertTrue(torch.allclose(roundtrip.params["lora_up"], el1.params["lora_up"]))

        # Mutate genome weights
        genome[0].lora_down.fill_(10.0)
        genome[0].lora_up.fill_(1.0)
        genome[1].lora_down.fill_(5.0)
        genome[1].lora_up.fill_(1.0)

        # Express back to new Grating
        new_grating = express_to_grating(genome, grating)

        # Verify new Grating holds the decoded (scaled) genotype values
        nodes = new_grating.nodes
        new_el1 = nodes["module1"]
        new_el2 = nodes["module2"]

        self.assertTrue(torch.allclose(new_el1.params["lora_down"], torch.full_like(new_el1.params["lora_down"], 10.0 * scales1[0])))
        self.assertTrue(torch.allclose(new_el1.params["lora_up"], torch.full_like(new_el1.params["lora_up"], scales1[1])))
        self.assertTrue(torch.allclose(new_el2.params["lora_down"], torch.full_like(new_el2.params["lora_down"], 5.0 * scales2[0])))

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

        # Wholesale layer crossover at cut point 2 (after layer 2)
        crossover = RandomNPointCrossover(num_cut_points=1, max_depth=1)
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

    def test_perturbation_gene_hierarchical_protocol(self) -> None:
        """Verify PerturbationGene works seamlessly with flatten_hierarchy and unflatten_hierarchy."""
        from neutral_selection.representation.hierarchy import flatten_hierarchy, unflatten_hierarchy

        gene = PerturbationGene(
            "layer_proto",
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
            True
        )

        # The activation (+1 for on) is a leaf after the weights; address and topology are not leaves
        leaves, treedef = flatten_hierarchy(gene)
        self.assertEqual(leaves, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0])
        self.assertEqual(treedef.total_leaves, 9)

        # Reconstruct with mutated leaves, including a switched-off activation
        mutated_leaves = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, -1.0]
        reconstructed = unflatten_hierarchy(mutated_leaves, treedef)

        self.assertIsInstance(reconstructed, PerturbationGene)
        self.assertEqual(reconstructed.address, "layer_proto")
        self.assertFalse(reconstructed.active)
        self.assertTrue(torch.equal(reconstructed.lora_down, torch.tensor([[10.0, 20.0], [30.0, 40.0]])))
        self.assertTrue(torch.equal(reconstructed.lora_up, torch.tensor([[50.0, 60.0], [70.0, 80.0]])))

    def test_hierarchical_lora_crossover(self) -> None:
        """Verify default HierarchicalCrossover splices LoRAGenome down into tensor features."""
        from neutral_selection.variation.recombination import HierarchicalCrossover

        # Layer 1: down (1, 2), up (2, 1) -> 4 elements
        # Layer 2: down (1, 2), up (2, 1) -> 4 elements
        # Total flat strand: 8 elements
        p1 = LoRAGenome([
            PerturbationGene("l1", torch.full((1, 2), 1.0), torch.full((2, 1), 1.0), True),
            PerturbationGene("l2", torch.full((1, 2), 2.0), torch.full((2, 1), 2.0), True),
        ])
        p2 = LoRAGenome([
            PerturbationGene("l1", torch.full((1, 2), 10.0), torch.full((2, 1), 10.0), False),
            PerturbationGene("l2", torch.full((1, 2), 20.0), torch.full((2, 1), 20.0), False),
        ])

        # Cut at linear index 2 (inside Layer 1's down tensor!)
        crossover = HierarchicalCrossover(cut_points=[2])
        child_a, child_b = crossover(p1, p2)

        self.assertIsInstance(child_a, LoRAGenome)
        self.assertIsInstance(child_b, LoRAGenome)
        self.assertEqual(len(child_a), 2)
        self.assertEqual(len(child_b), 2)

        # Child A: l1.down has [1.0, 1.0] from P1[:2], and l1.up has [10.0, 10.0] from P2[2:4]
        self.assertTrue(torch.equal(child_a[0].lora_down, torch.tensor([[1.0, 1.0]])))
        self.assertTrue(torch.equal(child_a[0].lora_up, torch.tensor([[10.0], [10.0]])))
        # Child A: l2 is 100% from P2
        self.assertTrue(torch.equal(child_a[1].lora_down, torch.full((1, 2), 20.0)))
        self.assertTrue(torch.equal(child_a[1].lora_up, torch.full((2, 1), 20.0)))

    def test_noise_is_uniform_in_genotype_and_scaled_at_expression(self) -> None:
        """Verify mutation adds plain noise to genotype values, and expression turns it into a width-appropriate step."""
        torch.manual_seed(0)
        strategy = get_lora_mutation_strategy(noise_scale=0.1, active_flips=0.0)

        # A fresh baseline element: Kaiming lora_down (rank=1, in=6144) and zero lora_up (out=1536)
        baseline = Grating()
        baseline.add_element(LoRAElement("l1", rank=1, alpha=1.0, in_features=6144, out_features=1536))
        genome = LoRAGenome.from_grating(baseline)
        child = mutate(genome, strategy)

        # Genotype steps have std noise_scale, whatever the layer's width
        self.assertAlmostEqual((child[0].lora_down - genome[0].lora_down).std().item(), 0.1, delta=0.01)
        self.assertAlmostEqual((child[0].lora_up - genome[0].lora_up).std().item(), 0.1, delta=0.01)

        # In weight space, lora_down steps are noise_scale times its 1/sqrt(fan_in) init scale
        parent_el = express_to_grating(genome, baseline).nodes["l1"]
        child_el = express_to_grating(child, baseline).nodes["l1"]
        down_step = child_el.params["lora_down"] - parent_el.params["lora_down"]
        self.assertAlmostEqual(down_step.std().item() * 6144 ** 0.5, 0.1, delta=0.01)

        # dW from a zero lora_up lands near noise_scale times the 1/sqrt(in) base weight scale
        dw_rms = (child_el.params["lora_up"] @ parent_el.params["lora_down"]).pow(2).mean().sqrt().item()
        self.assertAlmostEqual(dw_rms * 6144 ** 0.5, 0.1, delta=0.01)

        # Steps are fixed in genotype units: grown values do not get larger steps
        grown = LoRAGenome([LoRAGene("l1", torch.full((1, 4096), 50.0), torch.full((4096, 1), 50.0), True)])
        grown_child = mutate(grown, strategy)
        self.assertAlmostEqual((grown_child[0].lora_down - grown[0].lora_down).std().item(), 0.1, delta=0.01)

        # Half-precision tensors keep their dtype
        half = LoRAGenome([LoRAGene("l1", torch.ones(4, 4, dtype=torch.float16), torch.ones(4, 4, dtype=torch.float16), True)])
        self.assertEqual(mutate(half, strategy)[0].lora_down.dtype, torch.float16)

    def test_lora_genome_mutation_strategy(self) -> None:
        """Verify LoRAMutation perturbs every active gene at mutation_rate=1.0 and preserves the genome type."""
        genes = [
            PerturbationGene("l1", torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), True),
            PerturbationGene("l2", torch.zeros(2, 2), torch.zeros(2, 2), True)
        ]
        genome = LoRAGenome(genes)

        strategy = get_lora_mutation_strategy(noise_scale=0.1, active_flips=0.0, mutation_rate=1.0)

        mutated_genome = mutate(genome, strategy)
        self.assertIsInstance(mutated_genome, LoRAGenome)
        self.assertEqual(len(mutated_genome), 2)

        # First gene weights should be perturbed
        self.assertFalse(torch.equal(mutated_genome[0].lora_down, genome[0].lora_down))
        self.assertFalse(torch.equal(mutated_genome[0].lora_up, genome[0].lora_up))
        self.assertTrue(mutated_genome[0].active)

        # Second gene (zero init) should be initialized
        self.assertFalse(torch.all(mutated_genome[1].lora_down == 0.0))
        self.assertFalse(torch.all(mutated_genome[1].lora_up == 0.0))
        self.assertTrue(mutated_genome[1].active)

        # Parent tensors are never modified in place
        self.assertTrue(torch.all(genome[1].lora_up == 0.0))

    def test_inactive_genes_are_frozen(self) -> None:
        """Verify switched-off genes keep their weights, so switching back on restores the selected state."""
        random.seed(0)
        torch.manual_seed(0)
        dormant = PerturbationGene("l1", torch.full((1, 4), 0.5), torch.full((4, 1), 0.25), False)
        genome = LoRAGenome([dormant])

        # No flips: the dormant gene is shared untouched across generations
        child = genome
        for _ in range(10):
            child = mutate(child, get_lora_mutation_strategy(noise_scale=0.5, active_flips=0.0))
        self.assertIs(child[0].lora_down, dormant.lora_down)
        self.assertFalse(child[0].active)

        # Switching it back on restores exactly the weights it was switched off with
        revived = mutate(genome, get_lora_mutation_strategy(noise_scale=0.5, active_flips=1.0))
        self.assertTrue(revived[0].active)
        self.assertTrue(torch.equal(revived[0].lora_down, dormant.lora_down))
        self.assertTrue(torch.equal(revived[0].lora_up, dormant.lora_up))

        # A gene switched off in this step is frozen at its pre-step weights
        expressed = LoRAGenome([PerturbationGene("l1", torch.full((1, 4), 0.5), torch.full((4, 1), 0.25), True)])
        silenced = mutate(expressed, get_lora_mutation_strategy(noise_scale=0.5, active_flips=1.0))
        self.assertFalse(silenced[0].active)
        self.assertTrue(torch.equal(silenced[0].lora_down, expressed[0].lora_down))

        # Without freezing, a switched-off gene keeps drifting
        drifting = mutate(genome, get_lora_mutation_strategy(noise_scale=0.5, active_flips=0.0, freeze_inactive=False))
        self.assertFalse(drifting[0].active)
        self.assertFalse(torch.equal(drifting[0].lora_down, dormant.lora_down))

    def test_activation_is_inherited_with_its_weights(self) -> None:
        """Verify flattened crossover moves activations like any leaf, and mutation snaps blended ones to +/-1."""
        from neutral_selection.variation.recombination import ArithmeticCrossover

        on = LoRAGenome([PerturbationGene(f"l{i}", torch.zeros(1, 2), torch.zeros(2, 1), True) for i in range(4)])
        off = LoRAGenome([PerturbationGene(f"l{i}", torch.ones(1, 2), torch.ones(2, 1), False) for i in range(4)])

        # At gene granularity, each gene's activation travels with its weights from the same parent
        with unittest.mock.patch("random.sample", return_value=[2]):
            child_a, child_b = RandomNPointCrossover(num_cut_points=1, granularity="gene")(on, off)
        self.assertEqual([g.active for g in child_a], [True, True, False, False])
        self.assertEqual([g.active for g in child_b], [False, False, True, True])
        for gene in child_a:
            self.assertEqual(gene.active, bool(gene.lora_down.sum() == 0))

        # A blend leaves activations between +1 and -1; the next mutation snaps them back
        blended, _ = ArithmeticCrossover(alpha=0.75)(on, off)
        self.assertAlmostEqual(blended[0].activation.item(), 0.5)
        self.assertTrue(blended[0].active)
        snapped = mutate(blended, get_lora_mutation_strategy(noise_scale=0.0, active_flips=0.0))
        self.assertEqual([g.activation.item() for g in snapped], [1.0, 1.0, 1.0, 1.0])

    def test_active_flips_are_per_offspring(self) -> None:
        """Verify the expected number of flips per offspring is active_flips, independent of genome size."""
        random.seed(0)
        for gene_count in (20, 400):
            genome = LoRAGenome([
                PerturbationGene(f"l{i}", torch.ones(1, 2), torch.ones(2, 1), True) for i in range(gene_count)
            ])
            # mutation_rate=0 isolates flipping: weights stay untouched and genes are shared
            strategy = get_lora_mutation_strategy(noise_scale=0.1, active_flips=2.0, mutation_rate=0.0)
            trials = 500
            total_flips = 0
            for _ in range(trials):
                child = mutate(genome, strategy)
                total_flips += sum(1 for g in child if not g.active)
                for parent_gene, child_gene in zip(genome, child):
                    self.assertIs(child_gene.lora_down, parent_gene.lora_down)
            self.assertAlmostEqual(total_flips / trials, 2.0, delta=0.25)

        # More flips requested than genes: every gene flips
        small = LoRAGenome([PerturbationGene("l0", torch.ones(1, 2), torch.ones(2, 1), True)])
        child = mutate(small, get_lora_mutation_strategy(noise_scale=0.0, active_flips=5.0))
        self.assertFalse(child[0].active)

    def test_create_initial_population(self) -> None:
        """Verify LoRAGenome.create_initial_population generates expected copies and mutations."""
        grating = Grating()
        el = LoRAElement("l1", rank=2, alpha=1.0, in_features=2, out_features=2)
        grating.add_element(el)

        # Force the grating parameters to have specific values
        el.params["lora_down"].data.fill_(1.0)
        el.params["lora_up"].data.fill_(2.0)

        population = LoRAGenome.create_initial_population(
            base_grating=grating,
            population_size=3,
            noise_scale=0.1,
            active_flips=0.0
        )

        self.assertEqual(len(population), 3)
        for gen in population:
            self.assertIsInstance(gen, LoRAGenome)

        # All index positions (including 0, 1, and 2) are mutated
        self.assertFalse(torch.all(population[0][0].lora_down == 1.0))
        self.assertFalse(torch.all(population[1][0].lora_down == 1.0))
        self.assertFalse(torch.all(population[2][0].lora_down == 1.0))

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

    def test_genome_carries_topology(self) -> None:
        """Verify genes keep their element topology through save/load, hash it, and express without a baseline."""
        import tempfile
        import os
        from utils.uid import XXH3_64

        grating = Grating()
        grating.add_element(LoRAElement("lora_a", rank=2, alpha=4.0, in_features=6, out_features=3))
        grating.add_element(LoRAElement("lora_b", rank=1, alpha=1.0, in_features=5, out_features=2, reference_scale=0.5))
        genome = LoRAGenome.from_grating(grating)
        self.assertEqual(genome[0].topology["alpha"], 4.0)
        self.assertEqual(genome[1].topology["reference_scale"], 0.5)
        # Defaults are left out of the stored form
        self.assertEqual(genome[0].compact_topology(), {"alpha": 4.0})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "genome.safetensors")
            genome.save(path)
            loaded = LoRAGenome.load(path)
        self.assertEqual([g.topology for g in loaded], [g.topology for g in genome])

        # Expression needs no baseline, and rebuilds each element's kind, settings and shape
        expressed = express_to_grating(loaded)
        for address, original in grating.nodes.items():
            rebuilt = expressed.nodes[address]
            self.assertEqual(rebuilt.kernel_type, original.kernel_type)
            self.assertEqual(rebuilt.metadata["alpha"], original.metadata["alpha"])
            self.assertEqual(rebuilt.metadata["reference_scale"], original.metadata["reference_scale"])
            self.assertTrue(torch.allclose(rebuilt.params["lora_down"], original.params["lora_down"], atol=1e-6))
            self.assertEqual(rebuilt.params["lora_up"].shape, original.params["lora_up"].shape)

        # Topology is part of the content address
        uid_gen = XXH3_64()
        retuned = LoRAGenome([dataclasses.replace(genome[0], topology={**genome[0].topology, "alpha": 8.0}), genome[1]])
        self.assertNotEqual(uid_gen.from_state_dict(genome.get_state_dict()), uid_gen.from_state_dict(retuned.get_state_dict()))

    def test_legacy_genome_file_is_rejected(self) -> None:
        """Verify unversioned files, which store phenotype weights, fail to load until migrated."""
        import json
        import tempfile
        import os
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "legacy.safetensors")
            save_file(
                {"l1.lora_down": torch.randn(2, 8), "l1.lora_up": torch.randn(4, 2)},
                path,
                metadata={"addresses": json.dumps(["l1"]), "gene_metadata": json.dumps({"l1": {"active": True}})},
            )
            with self.assertRaisesRegex(ValueError, "legacy"):
                LoRAGenome.load(path)

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

            # Verify parameters were decoded with the baseline element's scales
            new_el = expressed_grating.nodes["module1"]
            down_scale, up_scale = lora_gene_scales(el1.params["lora_down"], el1.params["lora_up"])
            self.assertTrue(torch.allclose(new_el.params["lora_down"], (lora_down * down_scale).to(new_el.params["lora_down"].dtype)))
            self.assertTrue(torch.allclose(new_el.params["lora_up"], (lora_up * up_scale).to(new_el.params["lora_up"].dtype)))

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

    def test_scatter_evolution_api_structure(self) -> None:
        """Test `/wrap_individual` and `/scatter_evolution` API endpoints, verifying individual baseline wrapping, precursor audio linking, direct parent lineage, and scatter group compound hierarchy (no bundles)."""
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
            old_trigger = app_module.trigger_embedding_update
            app_module.param_graph = g
            app_module.engine_provider = provider
            app_module.trigger_embedding_update = MagicMock()
            
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
            el = LoRAElement("l1", rank=2, alpha=1.0, in_features=2, out_features=2)
            base_grating.add_element(el)
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

            # 4. Invoke /wrap_individual to turn precursor audio into baseline individual
            wrap_payload = {
                "precursor_id": "precursor_audio_test",
                "baseline_grating_id": "grating_test",
                "model_id": "model_test"
            }
            wrap_resp = client.post("/wrap_individual", json=wrap_payload)
            self.assertEqual(wrap_resp.status_code, 200)
            wrap_data = wrap_resp.get_json()
            self.assertTrue(wrap_data["success"])
            baseline_ind_id = wrap_data["node_id"]
            self.assertIsNotNone(baseline_ind_id)

            # Verify baseline individual in graph
            g_reloaded = ParameterGraph(tmp_dir)
            g_reloaded.load()
            base_ind = g_reloaded.get_element(baseline_ind_id)
            self.assertEqual(base_ind.type, "individual")
            self.assertEqual(base_ind.generation, 0)
            self.assertIsNone(base_ind.fitness)
            self.assertFalse(baseline_ind_id.startswith("individual_"))
            self.assertEqual(base_ind.id, base_ind.file.uid)
            self.assertTrue(base_ind.file.uid.endswith(".xxh3_64"))
            # Model binds_to individual
            self.assertTrue(g_reloaded.G.has_edge("model_test", baseline_ind_id))
            self.assertEqual(g_reloaded.G.get_edge_data("model_test", baseline_ind_id)["relation"], "binds_to")
            # Precursor edge
            self.assertTrue(g_reloaded.G.has_edge("precursor_audio_test", baseline_ind_id))
            self.assertEqual(g_reloaded.G.get_edge_data("precursor_audio_test", baseline_ind_id)["relation"], "precursor")

            # 5. Invoke /scatter_evolution on baseline individual
            payload = {
                "parents": [baseline_ind_id],
                "offspring_size": 3,
                "noise_scale": 0.05,
                # Single-gene genomes would flip on every child; flips are covered by unit tests
                "active_flips": 0.0,
                "generation_context": {
                    "operation": "generate",
                    "prompt": "Test Prompt"
                }
            }

            resp = client.post("/scatter_evolution", json=payload)
            self.assertEqual(resp.status_code, 202)
            
            res_data = resp.get_json()
            self.assertTrue(res_data["success"])
            self.assertIn("job_id", res_data)
            
            parent_job_id = res_data["job_id"]
            import time
            completed_data = None
            for _ in range(50):
                status_resp = client.get(f"/job_status/{parent_job_id}")
                if status_resp.status_code == 200:
                    status_data = status_resp.get_json()
                    if status_data.get("status") == "completed":
                        completed_data = status_data.get("result")
                        break
                    elif status_data.get("status") == "failed":
                        self.fail(f"Evolution scatter job failed: {status_data.get('error')}")
                time.sleep(0.01)
                
            self.assertIsNotNone(completed_data, "Job did not complete in time")
            res_data = completed_data

            # 6. Reload graph and verify elements & relationships
            g_reloaded = ParameterGraph(tmp_dir)
            g_reloaded.load()
            
            # Verify NO bundle node exists
            for nid in g_reloaded.G.nodes:
                el = g_reloaded.get_element(nid)
                self.assertNotEqual(el.type, "bundle")

            group_node = g_reloaded.get_element(res_data["group_id"])
            self.assertEqual(group_node.type, "group")
            self.assertEqual(len(group_node.member_ids), 3)

            # Check each individual member
            for ind_id in res_data["individual_ids"]:
                ind_node = g_reloaded.get_element(ind_id)
                self.assertEqual(ind_node.type, "individual")
                self.assertEqual(ind_node.generation, 1)
                self.assertFalse(ind_id.startswith("individual_"))
                self.assertEqual(ind_node.id, ind_node.file.uid)
                self.assertTrue(ind_node.file.uid.endswith(".xxh3_64"))
                # Parent compound must be the Group ID
                node_attrs = g_reloaded.G.nodes[ind_id]
                self.assertEqual(node_attrs.get("parent"), res_data["group_id"])
                
                # Direct parent individual edge: baseline_ind_id -> child ind_id
                self.assertTrue(g_reloaded.G.has_edge(baseline_ind_id, ind_id))
                self.assertEqual(g_reloaded.G.get_edge_data(baseline_ind_id, ind_id)["relation"], "parent")
                # Model binds_to edge
                self.assertTrue(g_reloaded.G.has_edge("model_test", ind_id))
                self.assertEqual(g_reloaded.G.get_edge_data("model_test", ind_id)["relation"], "binds_to")

            # Verify that registering a child artifact under an individual does NOT create an edge from the individual
            sample_ind_id = res_data["individual_ids"][0]
            exemplar_audio = Audio(
                id="exemplar_child_audio",
                name="Exemplar Audio",
                context={},
                file=Asset(path="mock_child.wav", uid="mock_child_uid", extension=".wav")
            )
            g_reloaded.add_element(exemplar_audio)
            g_reloaded.update_element(exemplar_audio.id, {"parent": sample_ind_id})
            self.assertFalse(g_reloaded.G.has_edge(sample_ind_id, exemplar_audio.id))

            # Verify that legacy parent->child edges are removed during graph load
            g_reloaded.G.add_edge(sample_ind_id, exemplar_audio.id, relation="source")
            self.assertTrue(g_reloaded.G.has_edge(sample_ind_id, exemplar_audio.id))
            g_reloaded.save()
            g_reloaded.load()
            self.assertFalse(g_reloaded.G.has_edge(sample_ind_id, exemplar_audio.id))

            # Verify that input edges into an exemplar generated from its individual are removed during graph load
            g_reloaded.update_element(exemplar_audio.id, {"context": {"individuals": [{"id": sample_ind_id, "strength": 1.0}]}})
            self.assertTrue(g_reloaded.is_generated_exemplar(exemplar_audio.id))
            g_reloaded.G.add_edge("model_test", exemplar_audio.id, type="model", relation="source")
            g_reloaded.save()
            g_reloaded.load()
            self.assertFalse(g_reloaded.G.has_edge("model_test", exemplar_audio.id))
            self.assertEqual(app_module.link_artifact_dependencies(g_reloaded.get_element(exemplar_audio.id), g_reloaded), 0)

            # Restore original globals
            app_module.param_graph = old_graph
            app_module.engine_provider = old_provider
            app_module.trigger_embedding_update = old_trigger

        finally:
            shutil.rmtree(tmp_dir)

    def test_scatter_evolution_genome_centric(self) -> None:
        """Test `/wrap_individual` with elements blueprint, `/scatter_evolution`, and `/express_individual` in a genome-centric manner without registering baseline grating nodes."""
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
            old_trigger = app_module.trigger_embedding_update
            app_module.param_graph = g
            app_module.engine_provider = provider
            app_module.trigger_embedding_update = MagicMock()
            
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

            # 3. Invoke /wrap_individual with elements config directly
            wrap_payload = {
                "model_id": "model_test",
                "precursor_id": "precursor_audio_test",
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
                ]
            }
            wrap_resp = client.post("/wrap_individual", json=wrap_payload)
            self.assertEqual(wrap_resp.status_code, 200)
            wrap_data = wrap_resp.get_json()
            self.assertTrue(wrap_data["success"])
            baseline_ind_id = wrap_data["node_id"]

            # 4. Invoke /scatter_evolution on baseline individual
            scatter_payload = {
                "parents": [baseline_ind_id],
                "offspring_size": 2,
                "noise_scale": 0.05,
                "active_flips": 1.0
            }
            resp = client.post("/scatter_evolution", json=scatter_payload)
            self.assertEqual(resp.status_code, 202)
            res_data = resp.get_json()
            self.assertTrue(res_data["success"])
            self.assertIn("job_id", res_data)
            
            parent_job_id = res_data["job_id"]
            import time
            completed_data = None
            for _ in range(50):
                status_resp = client.get(f"/job_status/{parent_job_id}")
                if status_resp.status_code == 200:
                    status_data = status_resp.get_json()
                    if status_data.get("status") == "completed":
                        completed_data = status_data.get("result")
                        break
                    elif status_data.get("status") == "failed":
                        self.fail(f"Evolution scatter job failed: {status_data.get('error')}")
                time.sleep(0.01)
                
            self.assertIsNotNone(completed_data, "Job did not complete in time")
            res_data = completed_data

            # 5. Verify no baseline grating was kept: the all-LoRA generation-0 grating lives in the genome
            g_reloaded = ParameterGraph(tmp_dir)
            g_reloaded.load()

            individual_id = res_data["individual_ids"][0]
            ind_node = g_reloaded.get_element(individual_id)
            self.assertEqual(ind_node.type, "individual")
            self.assertNotIn("baseline_elements", ind_node.context)
            self.assertNotIn("baseline_file_path", ind_node.context)
            self.assertIsNone(ind_node.baseline_grating_id)
            self.assertIsNone(g_reloaded.get_element(baseline_ind_id).baseline_grating_id)
            self.assertFalse([n for n in g_reloaded.G.nodes if g_reloaded.G.nodes[n].get("type") == "grating"])

            # 6. Verify /express_individual works on this individual from its genome alone
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
            app_module.trigger_embedding_update = old_trigger

        finally:
            shutil.rmtree(tmp_dir)

    def test_execute_operation_scatter(self) -> None:
        """Test `/execute_operation` endpoint with operation='wrap_individual' and operation='scatter'."""
        import tempfile
        import shutil
        from unittest.mock import AsyncMock
        from engine.engine_provider import EngineProvider
        import app as app_module
        from param_graph.graph import ParameterGraph
        from param_graph.elements.models.stylegan_element import StyleGANModel
        from param_graph.elements.artifacts.audio_element import Audio
        from param_graph.elements.base_elements import Asset

        tmp_dir = tempfile.mkdtemp()
        try:
            g = ParameterGraph(tmp_dir)
            provider = EngineProvider(data_root=tmp_dir)

            mock_engine = provider.get_engine()
            mock_engine.execute = AsyncMock(return_value="mock-job-id")

            old_graph = app_module.param_graph
            old_provider = app_module.engine_provider
            old_trigger = app_module.trigger_embedding_update
            app_module.param_graph = g
            app_module.engine_provider = provider
            app_module.trigger_embedding_update = MagicMock()
            
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
                context={"model_id": "model_test", "prompt": "mutant test sound"},
                file=Asset(path="mock_audio.wav", uid="mock_audio_uid", extension=".wav")
            )
            g.add_element(precursor_audio)
            g.save()

            # 3. Invoke /execute_operation with operation='wrap_individual'
            wrap_payload = {
                "operation": "wrap_individual",
                "execution_mode": "sync",
                "source_audio": "precursor_audio_test",
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
                ]
            }
            resp = client.post("/execute_operation", json=wrap_payload)
            self.assertEqual(resp.status_code, 200)
            wrap_data = resp.get_json()
            self.assertTrue(wrap_data["success"])
            baseline_ind_id = wrap_data["node_id"]

            # 4. Invoke /execute_operation with operation='scatter' on baseline individual
            payload = {
                "operation": "scatter",
                "execution_mode": "async",
                "parents": [baseline_ind_id],
                "offspring_size": 2,
                "noise_scale": 0.05,
                "active_flips": 1.0
            }
            resp = client.post("/execute_operation", json=payload)
            self.assertEqual(resp.status_code, 202)
            res_data = resp.get_json()
            self.assertTrue(res_data["success"])
            self.assertIn("job_id", res_data)
            
            parent_job_id = res_data["job_id"]
            import time
            completed_data = None
            for _ in range(50):
                status_resp = client.get(f"/job_status/{parent_job_id}")
                if status_resp.status_code == 200:
                    status_data = status_resp.get_json()
                    if status_data.get("status") == "completed":
                        completed_data = status_data.get("result")
                        break
                    elif status_data.get("status") == "failed":
                        self.fail(f"Evolution scatter job failed: {status_data.get('error')}")
                time.sleep(0.01)
                
            self.assertIsNotNone(completed_data, "Job did not complete in time")
            
            # Verify graph state
            g_reloaded = ParameterGraph(tmp_dir)
            g_reloaded.load()
            
            individual_id = completed_data["individual_ids"][0]
            ind_node = g_reloaded.get_element(individual_id)
            self.assertEqual(ind_node.type, "individual")
            self.assertEqual(ind_node.context.get("prompt"), "mutant test sound")

            # Restore original globals
            app_module.param_graph = old_graph
            app_module.engine_provider = old_provider
            app_module.trigger_embedding_update = old_trigger

        finally:
            shutil.rmtree(tmp_dir)

    def test_get_all_operations_endpoint(self) -> None:
        """Verify that /operations returns all operations and to_dict succeeds for every operation."""
        import app as app_module
        client = app_module.app.test_client()
        
        resp = client.get("/operations")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        ops = data.get("operations", [])
        self.assertGreater(len(ops), 0)
        
        op_names = {op["name"] for op in ops}
        self.assertIn("scatter", op_names)
        self.assertIn("gain", op_names)
        self.assertIn("normalize", op_names)
        
        for op in ops:
            self.assertIn("name", op)
            self.assertIn("category", op)
            self.assertIn("execution", op)
            self.assertIn("context_overrides", op)
            self.assertIn("form_config", op)

    def test_scatter_duplicate_individual_rejection(self) -> None:
        """Verify that scatter discards scattered individuals whose genome UIDs already exist in the graph."""
        import tempfile
        import shutil
        import os
        from param_graph.graph import ParameterGraph
        from param_graph.elements.models.stylegan_element import StyleGANModel
        from param_graph.elements.artifacts.individual_element import Individual
        from param_graph.elements.base_elements import Asset
        from operations.evolution.scatter import scatter_evolution_task
        from utils.uid import XXH3_64

        tmp_dir = tempfile.mkdtemp()
        try:
            g = ParameterGraph(tmp_dir)
            uid_gen = XXH3_64()

            model_node = StyleGANModel(
                id="model_mut_test",
                name="Scatter Test Model",
                context={},
                checkpoint=Asset(path="mock_path", uid="mock_uid", extension=".pt"),
                adapter="stylegan2"
            )
            g.add_element(model_node)

            output_dir = g.root / "generate"
            os.makedirs(output_dir, exist_ok=True)
            gene = LoRAGene("layer_fixed", torch.ones((2, 2)), torch.ones((2, 2)), True)
            genome = LoRAGenome([gene])
            parent_uid = uid_gen.from_state_dict(genome.get_state_dict())
            parent_genome_path = output_dir / f"{parent_uid}.safetensors"
            genome.save(str(parent_genome_path))

            parent_ind = Individual(
                id=parent_uid,
                name="Original Gen 0 Parent",
                file=Asset(path=str(parent_genome_path), uid=parent_uid, extension=".safetensors"),
                base_model_id="model_mut_test",
                generation=0,
                fitness=10.0,
                context={"model_id": "model_mut_test"}
            )
            g.add_element(parent_ind)
            g.save()

            # Run scatter task with 0 noise and 0 flip prob (yielding identical genome)
            scatter_evolution_task(
                parent_job_id="test_dup_scatter",
                parent_ids=[parent_uid],
                offspring_size=2,
                noise_scale=0.0,
                active_flips=0.0,
                mutation_rate=0.0,
                generation_context={},
                generate_exemplars=False,
                param_graph=g,
                uid_generator=uid_gen,
            )

            # Verify parent node was NOT overwritten
            parent_after = g.get_element(parent_uid)
            self.assertEqual(parent_after.name, "Original Gen 0 Parent")
            self.assertEqual(parent_after.generation, 0)
            self.assertIsNone(g.G.nodes[parent_uid].get("parent"))

            # Verify no self-loop edge exists
            self.assertFalse(g.G.has_edge(parent_uid, parent_uid))

        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()

