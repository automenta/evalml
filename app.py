import gradio as gr
import subprocess
import sys
import os
import pandas as pd
import database
import json
from threading import Thread
import tempfile
from comparison import run_experiment

def run_evaluation_from_file(config_file):
    """Runs an evaluation from an uploaded YAML file."""
    if config_file is None:
        return {"error": "No file uploaded."}

    config_path = config_file.name
    metrics = run_experiment(config_path, full_eval=True)

    if metrics:
        return metrics
    else:
        return {"error": "Failed to run evaluation. Check the logs."}

def run_discovery_process():
    """Runs the discovery.py script and yields its output line by line."""
    output_log = "Starting discovery process...\n"
    yield output_log

    command = [sys.executable, "discover.py", "--config", "config_discovery.yaml"]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        bufsize=1
    )

    for line in iter(process.stdout.readline, ''):
        output_log += line
        yield output_log

    process.stdout.close()
    process.wait()
    output_log += "\nDiscovery process finished."
    yield output_log

def get_discovery_results():
    """Loads and formats the discovery results from the database."""
    try:
        best_performers = database.get_best_performers(n=20, metric="perplexity", order="asc")
        if not best_performers:
            return pd.DataFrame(columns=["Model Name", "Perplexity", "Config"])

        results_data = []
        for result in best_performers:
            model_name = result.get('config', {}).get('evaluation', {}).get('model_name', 'N/A')
            perplexity = result.get('metrics', {}).get('perplexity', float('inf'))
            config_str = json.dumps(result.get('config', {}), indent=2)
            results_data.append([model_name, f"{perplexity:.4f}", config_str])

        df = pd.DataFrame(results_data, columns=["Model Name", "Perplexity", "Config"])
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=["Model Name", "Perplexity", "Config"])
    except Exception as e:
        print(f"Error loading discovery results: {e}")
        return pd.DataFrame(columns=["Model Name", "Perplexity", "Config"])

def get_model_choices():
    """Gets a list of model names from the discovery database."""
    try:
        results = database.load_all_results()
        model_names = [r.get('config', {}).get('evaluation', {}).get('model_name', f"Unnamed_{i}") for i, r in enumerate(results)]
        return gr.CheckboxGroup(choices=model_names, label="Select Models to Compare")
    except FileNotFoundError:
        return gr.CheckboxGroup(choices=[], label="Select Models to Compare (No results found)")

def compare_selected_models(selected_models):
    """Generates a comparison table for the selected models."""
    if not selected_models:
        return pd.DataFrame()

    try:
        all_results = database.load_all_results()
    except FileNotFoundError:
        return pd.DataFrame()

    comparison_data = {}
    metric_keys = set()

    for result in all_results:
        model_name = result.get('config', {}).get('evaluation', {}).get('model_name', 'N/A')
        if model_name in selected_models:
            metrics = result.get('metrics', {})
            sanitized_metrics = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in metrics.items()}
            comparison_data[model_name] = sanitized_metrics
            metric_keys.update(metrics.keys())

    df_data = []
    for metric in sorted(list(metric_keys)):
        row = {'Metric': metric}
        for model in selected_models:
            row[model] = comparison_data.get(model, {}).get(metric, 'N/A')
        df_data.append(row)

    return pd.DataFrame(df_data)


with gr.Blocks(title="Autonomous Model Discovery UI") as demo:
    gr.Markdown("# Autonomous Model Discovery UI")

    with gr.Tab("Discovery"):
        gr.Markdown("## 🚀 Autonomous Discovery of Optimal Models")
        gr.Markdown("Start the evolutionary search to discover novel, high-performing model architectures.")
        start_button = gr.Button("Start Discovery Process")
        output_log = gr.Textbox(label="Live Log", lines=20, interactive=False)
        results_table = gr.Dataframe(label="Top Performing Models", headers=["Model Name", "Perplexity", "Config"], interactive=False)

        start_button.click(
            fn=run_discovery_process,
            outputs=output_log
        ).then(
            fn=get_discovery_results,
            outputs=results_table
        )

    with gr.Tab("Evaluation"):
        gr.Markdown("## 🧪 Evaluate a Single Model")
        gr.Markdown("Upload a YAML configuration file to train and evaluate a specific model architecture.")
        config_file = gr.File(label="Upload YAML Config", file_types=[".yaml"])
        evaluate_button = gr.Button("Run Evaluation")
        evaluation_results = gr.JSON(label="Evaluation Metrics")

        evaluate_button.click(
            fn=run_evaluation_from_file,
            inputs=config_file,
            outputs=evaluation_results
        )

    with gr.Tab("Comparison"):
        gr.Markdown("## 📊 Compare Models")
        gr.Markdown("Select models from the discovery runs to compare their performance side-by-side.")
        model_choices = gr.CheckboxGroup(label="Select Models to Compare", interactive=True)
        compare_button = gr.Button("Compare Selected Models")
        comparison_table = gr.Dataframe(label="Comparison Results")

        compare_button.click(
            fn=compare_selected_models,
            inputs=model_choices,
            outputs=comparison_table
        )

    demo.load(get_discovery_results, None, results_table, every=5)
    demo.load(get_model_choices, None, model_choices, every=10)

if __name__ == "__main__":
    demo.launch()
