import argparse
import itertools
import json
import os
import copy
import yaml
from comparison import run_experiment

# Define the hyperparameter grid for the HREM model
# We use a simple dot notation for nested keys
PARAM_GRID = {
    "model.hidden_size": [128, 256],
    "model.num_layers_outer": [1, 2],
    "optimizer.lr": [1e-4, 5e-4],
}

def generate_configs(base_config_path, param_grid):
    """
    Generates a list of experiment configurations based on a parameter grid.
    """
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    configs = []
    keys, values = zip(*param_grid.items())

    for v_combination in itertools.product(*values):
        new_config = copy.deepcopy(base_config)
        config_name_parts = []

        for key, val in zip(keys, v_combination):
            # Update the nested dictionary
            parts = key.split('.')
            d = new_config
            for part in parts[:-1]:
                d = d[part]
            d[parts[-1]] = val
            config_name_parts.append(f"{parts[-1]}={val}")

        # Create a unique name for the model run
        model_name = base_config['evaluation']['model_name']
        new_config['evaluation']['model_name'] = f"{model_name} ({', '.join(config_name_parts)})"
        configs.append(new_config)

    return configs

def main():
    """
    Main function to run the hyperparameter sweep.
    """
    parser = argparse.ArgumentParser(description="Run a hyperparameter sweep for a given model.")
    parser.add_argument(
        "--config",
        type=str,
        default="config_hrem.yaml",
        help="Path to the base experiment configuration YAML file."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="hyperparam_results.json",
        help="Path to save the results of the hyperparameter sweep."
    )
    parser.add_argument(
        "--full-eval",
        action="store_true",
        help="Run full evaluation instead of a smoke test for each configuration."
    )
    parser.add_argument(
        "--baseline-config",
        type=str,
        default="config_baseline.yaml",
        help="Path to the baseline configuration YAML file for comparison."
    )
    args = parser.parse_args()

    print(f"--- Starting Hyperparameter Sweep ---")
    print(f"Base config: {args.config}")
    print(f"Output file: {args.output_file}")
    print(f"Full evaluation: {args.full_eval}")
    print(f"Parameter grid: {json.dumps(PARAM_GRID, indent=2)}")

    # Generate all configuration variants
    experiment_configs = generate_configs(args.config, PARAM_GRID)
    print(f"\nGenerated {len(experiment_configs)} configurations to test.")

    all_results = []

    for i, config in enumerate(experiment_configs):
        print(f"\n--- Running experiment {i+1}/{len(experiment_configs)} ---")
        print(f"Configuration: {config['evaluation']['model_name']}")

        # Create a temporary config file for this specific run
        temp_config_path = f"temp_config_{i}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)

        # Run the experiment
        metrics = run_experiment(temp_config_path, full_eval=args.full_eval)

        # Clean up the temporary config file
        os.remove(temp_config_path)

        if metrics:
            result_entry = {
                "config": config,
                "metrics": metrics
            }
            all_results.append(result_entry)
        else:
            print(f"Experiment {i+1} failed or returned no metrics.")

    # Save results to a JSON file
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\n\n--- Hyperparameter Sweep Finished ---")
    print(f"Results saved to {args.output_file}")

    # Run baseline experiment
    print("\n--- Running Baseline Experiment ---")
    baseline_metrics = run_experiment(args.baseline_config, full_eval=args.full_eval)
    if baseline_metrics:
        print("\n--- Baseline Results ---")
        print(json.dumps(baseline_metrics, indent=2))
    else:
        print("Baseline experiment failed or returned no metrics.")

    # Display a summary of the results
    if all_results:
        print("\n--- Results Summary (sorted by eval_loss) ---")
        sorted_results = sorted(all_results, key=lambda x: x['metrics'].get('eval_loss', float('inf')))

        # Display baseline results for comparison
        if baseline_metrics:
            loss = baseline_metrics.get('eval_loss', 'N/A')
            perplexity = baseline_metrics.get('perplexity', 'N/A')
            if isinstance(loss, float): loss = f"{loss:.4f}"
            if isinstance(perplexity, float): perplexity = f"{perplexity:.4f}"
            print(f"- {'Baseline':<50} | eval_loss={loss}, perplexity={perplexity}")
            print("-" * 80)


        for result in sorted_results:
            model_name = result['config']['evaluation']['model_name']
            loss = result['metrics'].get('eval_loss', 'N/A')
            perplexity = result['metrics'].get('perplexity', 'N/A')
            if isinstance(loss, float): loss = f"{loss:.4f}"
            if isinstance(perplexity, float): perplexity = f"{perplexity:.4f}"
            print(f"- {model_name:<50} | eval_loss={loss}, perplexity={perplexity}")

if __name__ == "__main__":
    main()
