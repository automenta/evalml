from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Config
from models.hrem import HREM
from wrappers import HREMWrapper
from config import ExperimentConfig

def get_model_and_optimizer(config: ExperimentConfig):
    """
    Returns the model and optimizer based on the configuration.
    """
    model_config = config.model
    optimizer_config = config.optimizer

    if model_config.model_type == 'gpt2':
        # Use a dictionary from the Pydantic model to initialize GPT2Config
        hf_config = GPT2Config(**model_config.dict())
        model = GPT2LMHeadModel(hf_config)
    elif model_config.model_type == 'hrem':
        # HREM expects a dictionary that includes batch_size and seq_len,
        # which are part of the evaluation config.
        hrem_config_dict = model_config.dict()
        hrem_config_dict['batch_size'] = config.evaluation.batch_size
        hrem_config_dict['seq_len'] = config.evaluation.block_size

        hrem_model_instance = HREM(hrem_config_dict)
        model = HREMWrapper(hrem_model_instance)
    else:
        raise ValueError(f"Unknown model type: {model_config.model_type}")

    if optimizer_config.name == 'adamw':
        optimizer = AdamW(model.parameters(), lr=optimizer_config.lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config.name}")

    return model, optimizer
