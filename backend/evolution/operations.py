"""Host-level evolutionary operations registry and definitions."""

def get_evolution_operations() -> list[dict]:
    """Returns the list of host-level evolutionary operations."""
    return [
        {
            "name": "mutate",
            "description": "Initialize a mutated population from this artifact",
            "category": "evolution",
            "execution": "queued",
            "initiator_types": ["audio"],
            "context_overrides": {},
            "form_config": [
                {
                    "name": "source_audio",
                    "type": "node",
                    "label": "Precursor Audio",
                    "filter": {"type": "audio"},
                    "required": True
                },
                {
                    "name": "population_size",
                    "type": "integer",
                    "label": "Population Size (N)",
                    "defaultValue": 10,
                    "min": 2,
                    "max": 100,
                    "allowSequence": False
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
                },
                {
                    "name": "lora_noise",
                    "type": "float",
                    "label": "LoRA Noise",
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
                    "defaultValue": 0.5,
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
            "initiator_types": ["group", "individual", "bundle"],
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
                    "name": "mutation_rate",
                    "type": "float",
                    "label": "Mutation Rate",
                    "defaultValue": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "allowSequence": False
                }
            ]
        }
    ]
