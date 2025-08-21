import torch
import torch.nn as nn
from config import EvolvedModelConfig, LayerConfig

class TransformerBlock(nn.Module):
    """
    A single block of a transformer model, using standard PyTorch components.
    """
    def __init__(self, layer_config: LayerConfig, hidden_size: int):
        super().__init__()
        self.n_head = layer_config.n_head
        self.hidden_size = hidden_size
        if self.hidden_size % self.n_head != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by n_head ({self.n_head})")

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.n_head,
            dim_feedforward=self.hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_layer(x)

class DynamicModel(nn.Module):
    """
    A dynamically constructed model based on EvolvedModelConfig.
    """
    def __init__(self, config: EvolvedModelConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)

        self.layers = nn.ModuleList(
            [TransformerBlock(layer_config, config.n_embd) for layer_config in config.layers]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, **batch):
        input_ids = batch.get('input_ids', batch.get('inputs'))
        labels = batch.get('labels', batch.get('targets'))

        if input_ids is None:
            raise KeyError("Input tensor not found in batch. Expected key 'input_ids' or 'inputs'.")

        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            logits_flat = shift_logits.view(-1, shift_logits.size(-1))
            labels_flat = shift_labels.view(-1)

            loss = nn.functional.cross_entropy(logits_flat, labels_flat, ignore_index=-1)

        return {'logits': logits, 'loss': loss}
