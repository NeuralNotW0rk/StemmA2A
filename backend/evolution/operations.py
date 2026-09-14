"""Host-level evolutionary operations registry and definitions."""

def get_evolution_operations() -> list[dict]:
    """Returns the list of host-level evolutionary operations."""
    return [
        {
            "name": "mutate",
            "description": "Initialize a mutated population from this artifact",
            "category": "evolution",
            "execution": "queued",
            "execution_mode": "async",  # backward compatibility alias
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
            "name": "reproduce",
            "description": "Breed next generation of offspring from this population",
            "category": "evolution",
            "execution": "queued",
            "execution_mode": "async",  # backward compatibility alias
            "initiator_types": ["bundle", "group"],
            "context_overrides": {},
            "form_config": [
                {
                    "name": "crossover_rate",
                    "type": "float",
                    "label": "Crossover Rate",
                    "defaultValue": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
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
                },
                {
                    "name": "elitism",
                    "type": "integer",
                    "label": "Elitism Count",
                    "defaultValue": 1,
                    "min": 0,
                    "max": 10,
                    "allowSequence": False
                }
            ]
        }
    ]
