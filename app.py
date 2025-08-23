import gradio as gr
from core import (
    run_discovery_process,
    get_discovery_results,
    run_evaluation_from_file,
    get_model_names,
    compare_selected_models,
)

def get_model_choices():
    """Gets a list of model names from the discovery database and returns a Gradio component."""
    model_names = get_model_names()
    return gr.CheckboxGroup(choices=model_names, label="Select Models to Compare")

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

    demo.load(get_discovery_results, outputs=results_table)
    timer_discovery = gr.Timer(5)
    timer_discovery.tick(get_discovery_results, outputs=results_table)

    demo.load(get_model_choices, outputs=model_choices)
    timer_models = gr.Timer(10)
    timer_models.tick(get_model_choices, outputs=model_choices)


if __name__ == "__main__":
    demo.launch()
