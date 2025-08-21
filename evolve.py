import random
import copy
from config import (
    ExperimentConfig,
    EvolvedModelConfig,
    LayerConfig,
    OptimizerConfig,
    DiscoveryConfig,
    IntSearchSpace,
    FloatSearchSpace,
    ChoiceSearchSpace,
)

def _sample_from_space(space):
    """Helper function to sample a value from a search space."""
    if isinstance(space, IntSearchSpace):
        return random.randint(space.min, space.max)
    if isinstance(space, FloatSearchSpace):
        return random.uniform(space.min, space.max)
    if isinstance(space, ChoiceSearchSpace):
        return random.choice(space.choices)
    raise TypeError(f"Unknown search space type: {type(space)}")

def create_initial_population(discovery_config: DiscoveryConfig, base_config: ExperimentConfig) -> list[ExperimentConfig]:
    """
    Creates an initial population of ExperimentConfig objects.
    """
    population = []
    arch_space = discovery_config.arch_space
    hyperparam_space = discovery_config.hyperparam_space

    for i in range(discovery_config.population_size):
        # Create a deep copy of the base config to modify
        new_config = copy.deepcopy(base_config)

        # --- Evolve Architectural Parameters ---
        num_layers = _sample_from_space(arch_space.n_layer)
        # All layers in a model must have the same n_head for RoPE to work correctly
        n_head = _sample_from_space(arch_space.n_head)
        layers = [LayerConfig(n_head=n_head) for _ in range(num_layers)]

        # Create a new EvolvedModelConfig
        model_config = EvolvedModelConfig(
            layers=layers,
            # Copy other necessary params from the base model config if they exist
            # For simplicity, we assume a base EvolvedModelConfig or similar structure
            vocab_size=new_config.model.vocab_size,
            n_positions=new_config.model.n_positions,
            n_embd=new_config.model.n_embd,
        )
        new_config.model = model_config

        # --- Evolve Hyperparameters ---
        lr = _sample_from_space(hyperparam_space.lr)
        new_config.optimizer = OptimizerConfig(lr=lr)

        # Update the model name to be unique
        new_config.evaluation.model_name = f"Evolved_Model_{i}"

        population.append(new_config)

    return population

def mutate(config: ExperimentConfig, discovery_config: DiscoveryConfig) -> ExperimentConfig:
    """
    Applies mutations to an ExperimentConfig object.
    Returns a new, mutated config.
    """
    mutated_config = copy.deepcopy(config)
    arch_space = discovery_config.arch_space
    hyperparam_space = discovery_config.hyperparam_space

    # Mutate learning rate
    if random.random() < discovery_config.mutation_rate:
        mutated_config.optimizer.lr = _sample_from_space(hyperparam_space.lr)

    # Mutate number of layers (add or remove a layer)
    if random.random() < discovery_config.mutation_rate and isinstance(mutated_config.model, EvolvedModelConfig):
        # Add a layer
        if len(mutated_config.model.layers) < arch_space.n_layer.max and random.random() < 0.5:
             if mutated_config.model.layers:
                existing_n_head = mutated_config.model.layers[0].n_head
                mutated_config.model.layers.append(LayerConfig(n_head=existing_n_head))
        # Remove a layer
        elif len(mutated_config.model.layers) > arch_space.n_layer.min:
            mutated_config.model.layers.pop(random.randrange(len(mutated_config.model.layers)))

    # Mutate n_head for all layers at once
    if random.random() < discovery_config.mutation_rate and isinstance(mutated_config.model, EvolvedModelConfig) and mutated_config.model.layers:
        new_n_head = _sample_from_space(arch_space.n_head)
        for layer in mutated_config.model.layers:
            layer.n_head = new_n_head

    return mutated_config

def crossover(parent1: ExperimentConfig, parent2: ExperimentConfig, discovery_config: DiscoveryConfig) -> ExperimentConfig:
    """
    Performs crossover between two parent configs to produce a child config.
    This implementation ensures the child has a consistent architecture.
    """
    child_config = copy.deepcopy(parent1)
    arch_space = discovery_config.arch_space

    # Crossover learning rate
    child_config.optimizer.lr = (parent1.optimizer.lr + parent2.optimizer.lr) / 2

    # Crossover architecture
    if isinstance(child_config.model, EvolvedModelConfig) and isinstance(parent2.model, EvolvedModelConfig):
        p1_model = parent1.model
        p2_model = parent2.model

        if not p1_model.layers or not p2_model.layers:
            return child_config # Cannot perform crossover if one parent has no layers

        # Crossover number of layers by averaging
        num_layers = (len(p1_model.layers) + len(p2_model.layers)) // 2
        num_layers = max(arch_space.n_layer.min, min(arch_space.n_layer.max, num_layers))

        # Crossover n_head by randomly choosing from one of the parents
        n_head = random.choice([p1_model.layers[0].n_head, p2_model.layers[0].n_head])

        child_config.model.layers = [LayerConfig(n_head=n_head) for _ in range(num_layers)]

    return child_config
