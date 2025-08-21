import yaml
import argparse
import copy
import random
from config import ExperimentConfig
from main import run_experiment
import evolve
import database

def main(config_path: str):
    """
    Main function to run the discovery process.
    """
    # 1. Load discovery configuration
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    base_config = ExperimentConfig(**config_dict)
    discovery_config = base_config.discovery
    if not discovery_config:
        raise ValueError("Discovery config not found in the provided YAML file.")

    # 2. Create initial population
    print("--- Creating Initial Population ---")
    population = evolve.create_initial_population(discovery_config, base_config)

    # 3. Run evolutionary loop
    for gen in range(discovery_config.num_generations):
        print(f"\n--- Generation {gen+1}/{discovery_config.num_generations} ---")

        # Evaluate population
        for i, individual_config in enumerate(population):
            print(f"\n--- Evaluating Individual {i+1}/{len(population)} (Gen {gen+1}) ---")
            individual_config.evaluation.model_name = f"Gen_{gen+1}_Ind_{i+1}"

            try:
                metrics = run_experiment(individual_config)
                database.save_result(individual_config, metrics)
            except Exception as e:
                print(f"Error evaluating individual: {e}")
                # Save failure result
                database.save_result(individual_config, {"error": str(e), "perplexity": float('inf')})

        # Breed next generation
        print("\n--- Breeding Next Generation ---")

        # Select parents from the best performers of all time
        num_parents = discovery_config.population_size // 2
        parents = database.get_best_performers(n=num_parents, metric="perplexity", order="asc")

        if not parents:
            print("No successful individuals to breed from. Re-initializing population.")
            next_population = evolve.create_initial_population(discovery_config, base_config)
        else:
            # Create next generation from parents
            next_population = []
            # Keep the best individual (elitism)
            best_config_dict = parents[0]['config']
            best_config = ExperimentConfig(**best_config_dict)
            next_population.append(best_config)

            # Fill the rest of the population with crossover and mutation
            while len(next_population) < discovery_config.population_size:
                p1_dict = random.choice(parents)['config']
                p2_dict = random.choice(parents)['config']
                p1_config = ExperimentConfig(**p1_dict)
                p2_config = ExperimentConfig(**p2_dict)

                # Crossover or mutation
                if random.random() < discovery_config.crossover_rate:
                    offspring = evolve.crossover(p1_config, p2_config, discovery_config)
                else:
                    offspring = evolve.mutate(p1_config, discovery_config)

                next_population.append(offspring)

        population = next_population

    print("\n--- Discovery Process Complete ---")
    best_overall = database.get_best_performers(n=1)[0]
    print(f"Best model found: {best_overall['config']['evaluation']['model_name']}")
    print(f"Best perplexity: {best_overall['metrics']['perplexity']}")
    print(f"Best config: {best_overall['config']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the architecture discovery process.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the discovery configuration YAML file."
    )
    args = parser.parse_args()
    main(args.config)
