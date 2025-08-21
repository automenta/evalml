# Autonomous Discovery of Optimal Models

This framework is designed for the autonomous discovery of optimal machine learning models for a given task. It goes beyond simple hyperparameter tuning and instead implements a form of **Neural Architecture Search (NAS)** to discover novel, high-performing model architectures.

## Methodology: Evolutionary Neural Architecture Search

The core of this framework is an evolutionary algorithm inspired by concepts from **Evolutionary Neural Architecture Search** and **Competitive Co-evolution**. The process works as follows:

1.  **Population Initialization**: The system starts with an initial "population" of diverse model architectures. This includes both standard baseline models (like Transformers) and randomly generated architectures.

2.  **Evaluation**: Each model in the population is trained and evaluated on the target task to measure its performance (e.g., perplexity).

3.  **Selection & Evolution**: The top-performing models are selected as "parents" for the next generation. New architectures are created by applying "mutations" to these parents. Mutations can include:
    - Adding or removing layers.
    - Changing layer types (e.g., swapping a Transformer block for a Mamba block).
    - Modifying layer-specific parameters (e.g., number of attention heads).
    - Adjusting global hyperparameters (e.g., learning rate).

4.  **Competitive Co-evolution for Fair Baselines**: A key feature of this methodology is the concept of a "fair race". By including baseline architectures in the evolving population, we ensure they are continuously optimized alongside the novel architectures. This prevents a situation where a new architecture appears superior only because the baseline was poorly tuned. Any discovered model must outperform a strong, co-evolved baseline, ensuring the discovered architectures are genuinely superior.

5.  **Iteration**: This cycle of evaluation, selection, and evolution repeats for many generations, progressively discovering more and more optimal models.

## Purpose

The primary goal of this framework is to automate the discovery of "drastically optimal algorithms and architectures" for small- and medium-scale models. By fearlessly applying high-risk, radical evolution, we aim for high-reward outcomes in model performance and efficiency.

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure the Discovery Process**:
    - Open `config_discovery.yaml`.
    - Adjust the `discovery` section to define your search space and parameters for the evolutionary search (e.g., `population_size`, `num_generations`).
    - The `evaluation` section can be configured for the evaluation of each individual. For a full run, set `smoke_test: false`.

3.  **Run the Discovery Pipeline**:
    ```bash
    python discover.py --config config_discovery.yaml
    ```
    The results of each experiment will be saved in `discovery_database.json`.
