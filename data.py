import os
import requests
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, default_data_collator
from transformers.utils import logging
from config import DataConfig

class ShakespeareDataset(Dataset):
    """A PyTorch Dataset for the Shakespeare dataset."""
    def __init__(self, tokenized_text, block_size):
        self.block_size = block_size
        self.examples = []
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):
            chunk = tokenized_text[i : i + block_size]
            self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # For language modeling, inputs and labels are the same.
        return {"input_ids": self.examples[i], "labels": self.examples[i]}

def setup_data(data_config: DataConfig, eval_config: 'EvalConfig'):
    """
    Sets up the data, including downloading, preprocessing, and creating dataloaders.
    """
    print("Setting up data...")
    if not os.path.exists(data_config.data_file):
        print(f"Downloading data from {data_config.url}...")
        r = requests.get(data_config.url)
        with open(data_config.data_file, 'w') as f:
            f.write(r.text)

    with open(data_config.data_file, 'r') as f:
        text = f.read()

    # For now, we assume a GPT-2 tokenizer. This could be made configurable.
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Temporarily suppress the warning about the long sequence length,
    # as the logic here handles chunking manually.
    logging.set_verbosity_error()
    tokenized_ids = tokenizer.encode(text)
    logging.set_verbosity_warning()  # Restore default verbosity

    tokenized_text = torch.tensor(tokenized_ids, dtype=torch.long)

    # Create datasets using the subset sizes from the config
    train_end = int(data_config.train_split * len(tokenized_text))
    train_data = tokenized_text[:data_config.train_subset]
    val_data = tokenized_text[train_end : train_end + data_config.eval_subset]

    train_dataset = ShakespeareDataset(train_data, eval_config.block_size)
    val_dataset = ShakespeareDataset(val_data, eval_config.block_size)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=eval_config.batch_size,
        collate_fn=default_data_collator
    )
    eval_dataloader = DataLoader(
        val_dataset,
        batch_size=eval_config.batch_size,
        collate_fn=default_data_collator
    )

    return train_dataloader, eval_dataloader, tokenizer
