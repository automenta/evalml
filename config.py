from pydantic import BaseModel, Field
from typing import Optional, Union, Literal

class DataConfig(BaseModel):
    """Configuration for data loading and preprocessing."""
    name: str = "tinyshakespeare"
    url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_file: str = "tinyshakespeare.txt"
    train_split: float = 0.9
    # Using small subsets for smoke testing
    train_subset: int = 10000
    eval_subset: int = 1000

class BaseModelConfig(BaseModel):
    """Base model for model configurations."""
    model_type: str

class GPT2ModelConfig(BaseModelConfig):
    """Configuration for GPT-2 models."""
    model_type: Literal["gpt2"] = "gpt2"
    vocab_size: int = 50257
    n_positions: int = 128
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12

class HREMModelConfig(BaseModelConfig):
    """Configuration for HREM models."""
    model_type: Literal["hrem"] = "hrem"
    # HREM specific parameters from eval_hrem.py
    vocab_size: int = 50257
    input_size: int
    hidden_size: int = 256
    output_size: int
    num_layers_inner: int = 2
    num_heads_inner: int = 4
    num_layers_outer: int = 2
    use_memory: bool = False
    forward_dtype: str = "float32"
    H_layers: int = 2
    L_layers: int = 2
    H_cycles: int = 1
    L_cycles: int = 1
    num_heads: int = 4
    expansion: float = 2.0
    pos_encodings: str = "rope"
    halt_max_steps: int = 1
    halt_exploration_prob: float = 0.0
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int = 1
    # External memory parameters (optional)
    m_loc: Optional[int] = None
    d_mem: Optional[int] = None
    top_k: Optional[int] = None
    sparse_addressing: bool = False
    use_location_addressing: bool = False

# --- Configuration for Evolvable Architectures ---

class LayerConfig(BaseModel):
    """Configuration for a single layer in a dynamic model."""
    layer_type: Literal["transformer_block"] = "transformer_block"
    n_head: int = 8
    # Other layer-specific parameters can be added here

class EvolvedModelConfig(BaseModelConfig):
    """Configuration for a dynamically evolved model."""
    model_type: Literal["evolved"] = "evolved"
    vocab_size: int = 50257
    n_positions: int = 128
    n_embd: int = 768
    layers: list[LayerConfig] = Field(default_factory=list)

# --- Configuration for the Search Space ---

class IntSearchSpace(BaseModel):
    """Defines a search space for an integer parameter."""
    min: int
    max: int

class FloatSearchSpace(BaseModel):
    """Defines a search space for a float parameter."""
    min: float
    max: float

class ChoiceSearchSpace(BaseModel):
    """Defines a search space for a categorical parameter."""
    choices: list[Union[str, int, float]]

class ArchSearchSpace(BaseModel):
    """Defines the search space for architectural parameters."""
    n_layer: IntSearchSpace = Field(default_factory=lambda: IntSearchSpace(min=2, max=12))
    n_head: ChoiceSearchSpace = Field(default_factory=lambda: ChoiceSearchSpace(choices=[4, 8, 12, 16]))
    # Can add more layer-specific search spaces here

class HyperparamSearchSpace(BaseModel):
    """Defines the search space for hyperparameters."""
    lr: FloatSearchSpace = Field(default_factory=lambda: FloatSearchSpace(min=1e-5, max=1e-3))

class DiscoveryConfig(BaseModel):
    """Configuration for the discovery (evolutionary search) process."""
    population_size: int = 20
    num_generations: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.5
    arch_space: ArchSearchSpace = Field(default_factory=ArchSearchSpace)
    hyperparam_space: HyperparamSearchSpace = Field(default_factory=HyperparamSearchSpace)


class OptimizerConfig(BaseModel):
    """Configuration for the optimizer."""
    name: Literal["adamw"] = "adamw"
    lr: float = 5e-5

class EvalConfig(BaseModel):
    """Configuration for the evaluation process."""
    block_size: int = 128
    batch_size: int = 16
    num_epochs: int = 1
    model_name: str
    smoke_test: bool = True

class ExperimentConfig(BaseModel):
    """Top-level configuration for an experiment."""
    data: DataConfig = Field(default_factory=DataConfig)
    model: Union[GPT2ModelConfig, HREMModelConfig, EvolvedModelConfig] = Field(..., discriminator='model_type')
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    evaluation: EvalConfig
    discovery: Optional[DiscoveryConfig] = None # Optional discovery config

    class Config:
        arbitrary_types_allowed = True
