import json
from typing import List, Dict, Any
from config import ExperimentConfig

DB_FILE = "discovery_database.json"

def save_result(config: ExperimentConfig, metrics: Dict[str, float]):
    """
    Saves a new experiment result to the database file.
    """
    try:
        results = load_all_results()
    except FileNotFoundError:
        results = []

    # Pydantic's .dict() method serializes the config to a dictionary
    new_result = {
        "config": config.dict(),
        "metrics": metrics
    }
    results.append(new_result)

    with open(DB_FILE, 'w') as f:
        json.dump(results, f, indent=4)

def load_all_results() -> List[Dict[str, Any]]:
    """
    Loads all results from the database file.
    """
    with open(DB_FILE, 'r') as f:
        return json.load(f)

def get_best_performers(n: int, metric: str = "perplexity", order: str = "asc") -> List[Dict[str, Any]]:
    """
    Loads all results and returns the top N performers based on a metric.
    'asc' for ascending order (lower is better), 'desc' for descending.
    """
    results = load_all_results()

    if not results:
        return []

    # Sort results
    reverse = (order == "desc")
    try:
        sorted_results = sorted(results, key=lambda r: r["metrics"][metric], reverse=reverse)
    except KeyError:
        raise KeyError(f"Metric '{metric}' not found in all results. Please check your database.")

    return sorted_results[:n]
