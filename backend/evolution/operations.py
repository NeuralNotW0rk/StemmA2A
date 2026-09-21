"""Host-level evolutionary operations registry and definitions."""

def get_evolution_operations() -> list[dict]:
    """Returns the list of host-level evolutionary operations."""
    return [
        {
            "name": "wrap_individual",
            "description": "Wrap this precursor artifact into a baseline evolutionary Individual",
            "category": "evolution",
            "execution": "immediate",
            "initiator_types": ["audio", "image", "grating", "latent"],
            "context_overrides": {},
            "form_config": [
                {
                    "name": "source_node",
                    "type": "node",
                    "label": "Precursor Artifact",
                    "filter": {"type": "audio"},
                    "required": True
                },
                {
                    "name": "lora_rank",
                    "type": "integer",
                    "label": "LoRA Rank",
                    "defaultValue": 1,
                    "min": 1,
                    "max": 128,
                    "allowSequence": False
                },
                {
                    "name": "lora_alpha",
                    "type": "float",
                    "label": "LoRA Alpha",
                    "defaultValue": 1.0,
                    "min": 0.1,
                    "step": 0.1,
                    "allowSequence": False
                }
            ]
        },
        {
            "name": "mutate",
            "description": "Mutate individual(s) to produce variant offspring",
            "category": "evolution",
            "execution": "queued",
            "initiator_types": ["individual", "group"],
            "context_overrides": {},
            "form_config": [
                {
                    "name": "parents",
                    "type": "node-list",
                    "label": "Target Individual(s)",
                    "filter": {"type": "individual"},
                    "required": True,
                    "minItems": 1
                },
                {
                    "name": "offspring_size",
                    "type": "integer",
                    "label": "Mutant Count (N)",
                    "defaultValue": 10,
                    "min": 1,
                    "max": 100,
                    "allowSequence": False
                },
                {
                    "name": "lora_noise",
                    "type": "float",
                    "label": "LoRA Perturbation Noise",
                    "defaultValue": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "allowSequence": False
                },
                {
                    "name": "active_flip_prob",
                    "type": "float",
                    "label": "Active Flip Probability",
                    "defaultValue": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "allowSequence": False
                },
                {
                    "name": "mutation_rate",
                    "type": "float",
                    "label": "Gene Mutation Rate",
                    "defaultValue": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "allowSequence": False
                }
            ]
        },
        {
            "name": "recombine",
            "description": "Recombine selected individuals into a new generation of variants",
            "category": "evolution",
            "execution": "queued",
            "initiator_types": ["group", "individual"],
            "context_overrides": {},
            "form_config": [
                {
                    "name": "parents",
                    "type": "node-list",
                    "label": "Parent Individuals",
                    "filter": {"type": "individual"},
                    "required": True,
                    "minItems": 1
                },
                {
                    "name": "offspring_size",
                    "type": "integer",
                    "label": "Offspring Count (N)",
                    "defaultValue": 10,
                    "min": 1,
                    "max": 100,
                    "allowSequence": False
                },
                {
                    "name": "crossover_type",
                    "type": "select",
                    "label": "Recombination Strategy",
                    "defaultValue": "two_point",
                    "options": [
                        {"label": "Two-Point Crossover (Multi-Tier Default)", "value": "two_point"},
                        {"label": "One-Point Crossover (Multi-Tier)", "value": "one_point"},
                        {"label": "Wholesale Layer Splicing (Depth 1)", "value": "layer_crossover"},
                        {"label": "Uniform Element Swap", "value": "uniform_crossover"},
                        {"label": "Continuous Weight Blend (BLX)", "value": "blend_crossover"}
                    ],
                    "allowSequence": False
                },
                {
                    "name": "selection_type",
                    "type": "select",
                    "label": "Selection Strategy",
                    "defaultValue": "tournament",
                    "options": [
                        {"label": "Tournament (Fitness-Biased)", "value": "tournament"},
                        {"label": "Uniform Random", "value": "uniform"}
                    ],
                    "allowSequence": False
                },
                {
                    "name": "crossover_prob",
                    "type": "float",
                    "label": "Crossover Probability",
                    "defaultValue": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "allowSequence": False
                },
                {
                    "name": "mutation_rate",
                    "type": "float",
                    "label": "Mutation Rate",
                    "defaultValue": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "allowSequence": False
                }
            ]
        }
    ]
