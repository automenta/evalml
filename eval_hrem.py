import subprocess
import os

def run_experiment(config_path):
    """Runs an experiment using the main.py script."""
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return

    print(f"\n--- Running experiment with config: {config_path} ---")

    command = ["python", "main.py", "--config", config_path]

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print("Error: main.py not found. Make sure you are in the correct directory.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the experiment: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Define the paths to the configuration files
    baseline_config_path = "config_baseline.yaml"
    hrem_config_path = "config_hrem.yaml"

    # Run the baseline experiment
    run_experiment(baseline_config_path)

    # Run the HREM experiment
    run_experiment(hrem_config_path)

    print("\n\n--- HREM Evaluation Script Finished ---")
    print("This script now orchestrates evaluations by calling main.py with different config files.")
