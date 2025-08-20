import yaml
import argparse
from config import ExperimentConfig
from data import setup_data
from registry import get_model_and_optimizer
from eval import train_and_evaluate

def main(config_path: str):
    """
    Main function to run an evaluation experiment.
    """
    # 1. Load configuration from YAML file
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # 2. Parse the configuration using Pydantic models
    try:
        config = ExperimentConfig(**config_dict)
    except Exception as e:
        print(f"Error parsing configuration: {e}")
        return

    # 3. Setup data
    train_dataloader, eval_dataloader, tokenizer = setup_data(config.data, config.evaluation)

    # 4. Get model and optimizer
    # Update vocab_size in model config from tokenizer
    config.model.vocab_size = tokenizer.vocab_size
    if hasattr(config.model, 'output_size'):
        config.model.output_size = tokenizer.vocab_size
    if hasattr(config.model, 'input_size'):
        config.model.input_size = tokenizer.vocab_size

    model, optimizer = get_model_and_optimizer(config)

    # 5. Run training and evaluation
    print(f"\n--- Evaluating Model: {config.evaluation.model_name} ---")
    metrics = train_and_evaluate(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        num_epochs=config.evaluation.num_epochs,
        model_name=config.evaluation.model_name,
        smoke_test=config.evaluation.smoke_test
    )

    # 6. Print results
    print(f"\n--- Results for {config.evaluation.model_name} ---")
    print(f"Metrics: {metrics}")
    print("\nEvaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a model evaluation experiment.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment configuration YAML file."
    )
    args = parser.parse_args()
    main(args.config)
