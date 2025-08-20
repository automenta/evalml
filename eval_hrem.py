import subprocess
import os
import sys
import yaml
import re
import json

def run_experiment(config_path):
    """
    Runs an experiment using main.py and returns the metrics.
    """
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return None

    print(f"\n--- Running experiment with config: {config_path} ---")
    command = [sys.executable, "main.py", "--config", config_path]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        print(result.stdout)
        if result.stderr:
            print("--- STDERR ---")
            print(result.stderr)

        # Find the metrics line in the output and parse it
        metrics_match = re.search(r"Metrics: (\{.*\})", result.stdout)
        if metrics_match:
            # The output from printing a dict uses single quotes, which is not valid JSON.
            # Replace single quotes with double quotes to parse with the json module.
            metrics_str = metrics_match.group(1).replace("'", '"')
            try:
                metrics_dict = json.loads(metrics_str)
                return metrics_dict
            except json.JSONDecodeError as e:
                print(f"Error parsing metrics JSON: {e}")
                print(f"Metrics string was: {metrics_str}")
                return None
        else:
            print("Could not find metrics in the output.")
            return None

    except FileNotFoundError:
        print("Error: main.py not found. Make sure you are in the correct directory.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the experiment: {e}")
        print("\n--- STDOUT ---")
        print(e.stdout)
        print("\n--- STDERR ---")
        print(e.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def display_configs(baseline_config, hrem_config):
    """Displays the configurations side-by-side."""
    print("\n\n--- Experiment Configurations ---")
    print("\n--- Baseline Configuration (config_baseline.yaml) ---")
    print(yaml.dump(baseline_config, default_flow_style=False))
    print("\n--- HREM Configuration (config_hrem.yaml) ---")
    print(yaml.dump(hrem_config, default_flow_style=False))

def display_comparison(baseline_results, hrem_results):
    """Displays a comparison of the results."""
    print("\n\n--- Experiment Results Comparison ---")
    header = f"{'Metric':<20} | {'Baseline':<20} | {'HREM Novel':<20}"
    print(header)
    print("-" * len(header))

    if not baseline_results or not hrem_results:
        print("Could not retrieve results for one or both experiments.")
        return

    all_keys = sorted(list(set(baseline_results.keys()) | set(hrem_results.keys())))

    for key in all_keys:
        baseline_val = baseline_results.get(key, "N/A")
        hrem_val = hrem_results.get(key, "N/A")

        if isinstance(baseline_val, float):
            baseline_val = f"{baseline_val:.4f}"
        if isinstance(hrem_val, float):
            hrem_val = f"{hrem_val:.4f}"

        row = f"{key:<20} | {str(baseline_val):<20} | {str(hrem_val):<20}"
        print(row)
    print("-" * len(header))

if __name__ == "__main__":
    baseline_config_path = "config_baseline.yaml"
    hrem_config_path = "config_hrem.yaml"

    # Load configs to display them
    try:
        with open(baseline_config_path, 'r') as f:
            baseline_config = yaml.safe_load(f)
        with open(hrem_config_path, 'r') as f:
            hrem_config = yaml.safe_load(f)

        display_configs(baseline_config, hrem_config)

    except FileNotFoundError as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)

    # Run experiments and get results
    baseline_results = run_experiment(baseline_config_path)
    hrem_results = run_experiment(hrem_config_path)

    # Display comparison
    display_comparison(baseline_results, hrem_results)

    print("\n\n--- HREM Evaluation Script Finished ---")
