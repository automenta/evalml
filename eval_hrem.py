import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import requests
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, default_data_collator
from models.hrem import HREM
from eval import train_and_evaluate
import torch.nn as nn

# --- Configuration ---
BLOCK_SIZE = 128
BATCH_SIZE = 16 # Smaller batch size for local testing
NUM_EPOCHS = 1 # For smoke test
MODEL_NAME = 'gpt2'
DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
DATA_FILE = 'tinyshakespeare.txt'

# --- 1. Load and Preprocess Data ---
class ShakespeareDataset(Dataset):
    def __init__(self, tokenized_text, block_size):
        self.block_size = block_size
        self.examples = []
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):
            chunk = tokenized_text[i : i + block_size]
            self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # For language modeling, inputs and labels are the same
        return {"input_ids": self.examples[i], "labels": self.examples[i]}

def setup_data():
    print("Loading and preprocessing data...")
    if not os.path.exists(DATA_FILE):
        r = requests.get(DATA_URL)
        with open(DATA_FILE, 'w') as f:
            f.write(r.text)

    with open(DATA_FILE, 'r') as f:
        text = f.read()

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_text = tokenizer.encode(text, return_tensors='pt')[0]

    # Create datasets
    n = len(tokenized_text)
    train_end = int(0.9 * n)
    # Use a small subset for the smoke test
    train_data = tokenized_text[:10000]
    val_data = tokenized_text[train_end : train_end + 1000]

    train_dataset = ShakespeareDataset(train_data, BLOCK_SIZE)
    val_dataset = ShakespeareDataset(val_data, BLOCK_SIZE)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=default_data_collator)
    eval_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=default_data_collator)

    return train_dataloader, eval_dataloader, tokenizer

# --- 2. HREM Model Wrapper ---
class HREMWrapper(nn.Module):
    def __init__(self, hrem_model, vocab_size):
        super().__init__()
        self.hrem = hrem_model
        # The HREM model has its own lm_head.

    def forward(self, input_ids, labels=None, **kwargs):
        batch = {'inputs': input_ids, 'puzzle_identifiers': torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)}

        if not hasattr(self, 'carry') or self.carry[0].inner_carry.z_H.shape[0] != input_ids.shape[0]:
             self.carry = self.hrem.initial_carry(batch)

        (self.carry, outputs) = self.hrem.forward(self.carry, batch)

        logits = outputs['logits']

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        from types import SimpleNamespace
        return SimpleNamespace(loss=loss, logits=logits)


# --- 3. Main Execution ---
if __name__ == "__main__":
    # Setup data
    train_dataloader, eval_dataloader, tokenizer = setup_data()
    vocab_size = tokenizer.vocab_size

    # --- Baseline Model (GPT-2) ---
    print("\n--- Evaluating Baseline: GPT-2 ---")
    gpt2_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=BLOCK_SIZE,
        n_embd=768,
        n_layer=12,
        n_head=12,
    )
    baseline_model = GPT2LMHeadModel(gpt2_config)
    baseline_optimizer = AdamW(baseline_model.parameters(), lr=5e-5)

    baseline_metrics = train_and_evaluate(
        model=baseline_model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=baseline_optimizer,
        num_epochs=NUM_EPOCHS,
        model_name="GPT-2 Baseline"
    )

    # --- Novel Model (HREM) ---
    print("\n--- Evaluating Novel Model: HREM ---")

    hrem_config = {
        'batch_size': BATCH_SIZE,
        'seq_len': BLOCK_SIZE,
        'vocab_size': vocab_size,
        'input_size': vocab_size,
        'hidden_size': 256,
        'output_size': vocab_size,
        'num_layers_inner': 2,
        'num_heads_inner': 4,
        'num_layers_outer': 2,
        'use_memory': False,
        'forward_dtype': 'float32',
        'H_layers': 2,
        'L_layers': 2,
        'H_cycles': 1,
        'L_cycles': 1,
        'num_heads': 4,
        'expansion': 2.0,
        'pos_encodings': 'rope',
        'halt_max_steps': 1,
        'halt_exploration_prob': 0.0,
        'puzzle_emb_ndim': 0,
        'num_puzzle_identifiers': 1
    }

    from types import SimpleNamespace
    hrem_model_instance = HREM(hrem_config)
    novel_model = HREMWrapper(hrem_model_instance, vocab_size)
    novel_optimizer = AdamW(novel_model.parameters(), lr=1e-4)

    # The HREM wrapper is still a major simplification.
    # The forward pass of HREM is stateful and complex.
    # A simple loop might not be enough.
    # For now, let's see if it runs.
    try:
        novel_metrics = train_and_evaluate(
            model=novel_model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            optimizer=novel_optimizer,
            num_epochs=NUM_EPOCHS,
            model_name="HREM Novel"
        )
    except Exception as e:
        print(f"Error training HREM model: {e}")
        novel_metrics = {'eval_loss': 'error', 'perplexity': 'error', 'final_perplexity': 'error'}


    # --- 4. Print Comparison ---
    print("\n\n--- SMOKE TEST RESULTS ---")
    print(f"Baseline (GPT-2) Metrics: {baseline_metrics}")
    print(f"Novel (HREM) Metrics:    {novel_metrics}")
    print("\nSmoke test complete.")
