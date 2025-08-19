import torch
import torch.nn as nn
from types import SimpleNamespace
from models.hrem import HREM

class HREMWrapper(nn.Module):
    """
    A wrapper for the HREM model to make it compatible with the evaluation script.
    """
    def __init__(self, hrem_model: HREM):
        super().__init__()
        self.hrem = hrem_model
        self.carry = None

    def forward(self, input_ids, labels=None, **kwargs):
        batch = {
            'inputs': input_ids,
            'puzzle_identifiers': torch.zeros(
                input_ids.shape[0],
                dtype=torch.long,
                device=input_ids.device
            )
        }

        # Initialize carry state if it's not present or if batch size changes
        if self.carry is None or self.carry[0].inner_carry.z_H.shape[0] != input_ids.shape[0]:
            self.carry = self.hrem.initial_carry(batch)

        (self.carry, outputs) = self.hrem.forward(self.carry, batch)
        logits = outputs['logits']

        loss = None
        if labels is not None:
            # Standard language model loss calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return SimpleNamespace(loss=loss, logits=logits)
