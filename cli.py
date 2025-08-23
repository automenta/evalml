import argparse
import pandas as pd
from core import (
    run_discovery_process,
    get_discovery_results,
    run_evaluation_from_file,
    get_model_names,
    compare_selected_models,
)

def main():
    parser = argparse.ArgumentParser(description="A command-line interface for the model discovery tool.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Discover command
    parser_discover = subparsers.add_parser("discover", help="Run the discovery process.")

    # Results command
    parser_results = subparsers.add_parser("results", help="Show the top performing models from the discovery process.")

    # Evaluate command
    parser_evaluate = subparsers.add_parser("evaluate", help="Evaluate a model from a config file.")
    parser_evaluate.add_argument("config_file", help="Path to the YAML config file.")

    # List models command
    parser_list_models = subparsers.add_parser("list-models", help="List all available models for comparison.")

    # Compare command
    parser_compare = subparsers.add_parser("compare", help="Compare selected models.")
    parser_compare.add_argument("models", nargs="+", help="Names of the models to compare.")

    args = parser.parse_args()

    if args.command == "discover":
        for line in run_discovery_process():
            print(line, end="")

    elif args.command == "results":
        results_df = get_discovery_results()
        print(results_df.to_string())

    elif args.command == "evaluate":
        # We need to simulate the File object that gradio uses
        class MockGradioFile:
            def __init__(self, name):
                self.name = name

        mock_file = MockGradioFile(args.config_file)
        metrics = run_evaluation_from_file(mock_file)
        print(metrics)

    elif args.command == "list-models":
        model_names = get_model_names()
        if model_names:
            print("Available models:")
            for model in model_names:
                print(f"- {model}")
        else:
            print("No models found.")

    elif args.command == "compare":
        comparison_df = compare_selected_models(args.models)
        print(comparison_df.to_string())


if __name__ == "__main__":
    main()
