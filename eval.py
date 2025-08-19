import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import os
import requests
from torch.utils.data import Dataset, DataLoader
import argparse
import math
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import uuid

# Hyperparameters and constants
DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
DATA_FILE = 'input.txt'
BLOCK_SIZE = 128
N_EMBD = 64
N_HEAD = 4
N_LAYER = 4
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200
GENERATE_TOKENS = 500
TRAIN_SPLIT = 0.9
VAL_SPLIT = 0.05

class Task:
    def __init__(self, train_data, val_data, test_data, vocab_size, stoi=None, itos=None, tokenizer=None):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.vocab_size = vocab_size
        self.stoi = stoi
        self.itos = itos
        self.tokenizer = tokenizer

    def get_data_loaders(self):
        raise NotImplementedError("Must implement get_data_loaders in subclass")

    def generate_text(self, model, device):
        raise NotImplementedError("Must implement generate_text in subclass")

    def get_param_count(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def estimate_loss(self, model, loader, device):
        model.eval()
        losses = torch.zeros(EVAL_ITERS)
        with torch.no_grad():
            for k in range(EVAL_ITERS):
                xb, yb = next(iter(loader))
                xb, yb = xb.to(device), yb.to(device)
                _, loss = model(xb, yb)
                losses[k] = loss.item()
        model.train()
        return losses.mean().item()

    def train_and_evaluate(self, model, model_name, device):
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        train_loader, val_loader, test_loader = self.get_data_loaders()
        start_time = time.time()
        peak_memory = 0
        train_losses = []
        val_losses = []

        num_batches = len(train_loader)
        for epoch in range(EPOCHS):
            model.train()
            epoch_train_loss = 0
            for batch_idx, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(device), yb.to(device)
                logits, loss = model(xb, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

                if device == 'cuda':
                    current_mem = torch.cuda.max_memory_allocated() / 1e6
                    peak_memory = max(peak_memory, current_mem)

                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == num_batches:
                    print(f"{model_name} - Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx+1}/{num_batches}, Train Loss: {loss.item():.4f}")

            avg_train_loss = epoch_train_loss / num_batches
            train_losses.append(avg_train_loss)
            val_loss = self.estimate_loss(model, val_loader, device)
            val_losses.append(val_loss)
            print(f"{model_name} - Epoch {epoch+1}/{EPOCHS}, Avg Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        train_time = time.time() - start_time
        test_loss = self.estimate_loss(model, test_loader, device)
        perplexity = math.exp(test_loss)
        generated_text = self.generate_text(model, device)

        return {
            'param_count': self.get_param_count(model),
            'train_time': train_time,
            'infer_time': generated_text['infer_time'],
            'peak_memory_MB': peak_memory if device == 'cuda' else 'N/A (CPU)',
            'test_loss': test_loss,
            'perplexity': perplexity,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'generated_text': generated_text['text']
        }

    def print_comparison(self, metrics_dict, task_name):
        print(f"\nPerformance Comparison for {task_name}:\n")
        headers = ["Metric"] + list(metrics_dict.keys())
        data = [
            ("Parameter Count", [m['param_count'] for m in metrics_dict.values()]),
            ("Training Time (s)", [f"{m['train_time']:.2f}" for m in metrics_dict.values()]),
            ("Inference Time (s/token)", [f"{m['infer_time']:.4f}" for m in metrics_dict.values()]),
            ("Peak Memory (MB)", [m['peak_memory_MB'] for m in metrics_dict.values()]),
            ("Test Loss", [f"{m['test_loss']:.4f}" for m in metrics_dict.values()]),
            ("Perplexity", [f"{m['perplexity']:.2f}" for m in metrics_dict.values()]),
            ("Best Val Loss", [f"{min(m['val_losses']):.4f}" for m in metrics_dict.values()])
        ]

        col_widths = [max(len(str(row[0])) for row in data)]
        for i in range(1, len(headers)):
            col_widths.append(max(len(str(row[i][j])) for row in data for j in range(len(metrics_dict)) if j == i - 1))
        print(" | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers))))
        print("-" * (sum(col_widths) + 4 * len(headers) - 2))
        for row in data:
            row_values = [row[0]] + [str(item) for item in row[1]]
            print(" | ".join(f"{row_values[i]:<{col_widths[i]}}" for i in range(len(row_values))))

        print(f"\nTraining Loss Curves for {task_name}:")
        headers = ["Epoch"] + [f"{name} Train | {name} Val" for name in metrics_dict.keys()]
        col_widths = [5] + [13 for _ in range(len(metrics_dict) * 2)]
        print(" | ".join(f"{header:<{col_widths[i]}}" for i, header in enumerate(headers)))
        print("-" * (sum(col_widths) + 4 * len(headers) - 2))
        for epoch in range(EPOCHS):
            row = [f"{epoch+1}"]
            for name in metrics_dict:
                row.append(f"{metrics_dict[name]['train_losses'][epoch]:.4f}")
                row.append(f"{metrics_dict[name]['val_losses'][epoch]:.4f}")
            print(" | ".join(f"{item:<{col_widths[i]}}" for i, item in enumerate(row)))

        for name, metrics in metrics_dict.items():
            print(f"\n{name} Generated Text ({task_name}):")
            print(metrics['generated_text'])

class CharLanguageModelingTask(Task):
    class CharDataset(Dataset):
        def __init__(self, data, block_size):
            self.data = data
            self.block_size = block_size

        def __len__(self):
            return len(self.data) - self.block_size

        def __getitem__(self, idx):
            chunk = self.data[idx:idx + self.block_size + 1]
            x = chunk[:-1]
            y = chunk[1:]
            return x, y

    def get_data_loaders(self):
        return (
            DataLoader(self.CharDataset(self.train_data, BLOCK_SIZE), batch_size=BATCH_SIZE, shuffle=True, drop_last=True),
            DataLoader(self.CharDataset(self.val_data, BLOCK_SIZE), batch_size=BATCH_SIZE, drop_last=True),
            DataLoader(self.CharDataset(self.test_data, BLOCK_SIZE), batch_size=BATCH_SIZE, drop_last=True)
        )

    def generate_text(self, model, device):
        start_time = time.time()
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(GENERATE_TOKENS):
                logits, _ = model(idx[:, -BLOCK_SIZE:])
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        infer_time = (time.time() - start_time) / GENERATE_TOKENS
        generated = ''.join([self.itos[int(i)] for i in idx[0]])
        return {'text': generated, 'infer_time': infer_time}

class TokenLanguageModelingTask(Task):
    class TokenDataset(Dataset):
        def __init__(self, data, block_size):
            self.data = data
            self.block_size = block_size

        def __len__(self):
            return len(self.data) - self.block_size

        def __getitem__(self, idx):
            chunk = self.data[idx:idx + self.block_size + 1]
            x = chunk[:-1]
            y = chunk[1:]
            return x, y

    def get_data_loaders(self):
        return (
            DataLoader(self.TokenDataset(self.train_data, BLOCK_SIZE), batch_size=BATCH_SIZE, shuffle=True, drop_last=True),
            DataLoader(self.TokenDataset(self.val_data, BLOCK_SIZE), batch_size=BATCH_SIZE, drop_last=True),
            DataLoader(self.TokenDataset(self.test_data, BLOCK_SIZE), batch_size=BATCH_SIZE, drop_last=True)
        )

    def generate_text(self, model, device):
        start_time = time.time()
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(GENERATE_TOKENS):
                logits, _ = model(idx[:, -BLOCK_SIZE:])
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        infer_time = (time.time() - start_time) / GENERATE_TOKENS
        generated = self.tokenizer.decode(idx[0].tolist(), skip_special_tokens=True)
        return {'text': generated, 'infer_time': infer_time}

def get_char_data():
    if not os.path.exists(DATA_FILE):
        try:
            r = requests.get(DATA_URL, timeout=10)
            r.raise_for_status()
            with open(DATA_FILE, 'w') as f:
                f.write(r.text)
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download dataset: {e}")
    with open(DATA_FILE, 'r') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    n = len(data)
    train_end = int(TRAIN_SPLIT * n)
    val_end = train_end + int(VAL_SPLIT * n)
    return data[:train_end], data[train_end:val_end], data[val_end:], vocab_size, stoi, itos

def get_token_data():
    if not os.path.exists(DATA_FILE):
        try:
            r = requests.get(DATA_URL, timeout=10)
            r.raise_for_status()
            with open(DATA_FILE, 'w') as f:
                f.write(r.text)
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download dataset: {e}")
    with open(DATA_FILE, 'r') as f:
        text = f.read()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokens = tokenizer.encode(text, return_tensors='pt')[0]
    n = len(tokens)
    train_end = int(TRAIN_SPLIT * n)
    val_end = train_end + int(VAL_SPLIT * n)
    return tokens[:train_end], tokens[train_end:val_end], tokens[val_end:], tokenizer

class LanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.block_size = BLOCK_SIZE

    def forward(self, idx, targets=None):
        raise NotImplementedError("Forward method must be implemented by subclass")

class BaselineTransformer(LanguageModel):
    def __init__(self, vocab_size):
        super().__init__(vocab_size)
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=BLOCK_SIZE,
            n_embd=N_EMBD,
            n_layer=N_LAYER,
            n_head=N_HEAD,
        )
        self.model = GPT2LMHeadModel(config)

    def forward(self, idx, targets=None):
        outputs = self.model(idx, labels=targets)
        logits = outputs.logits
        loss = outputs.loss if targets is not None else None
        return logits, loss

class NovelAlgorithm(LanguageModel):
    def __init__(self, vocab_size):
        super().__init__(vocab_size)
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=BLOCK_SIZE,
            n_embd=N_EMBD,
            n_layer=N_LAYER,
            n_head=N_HEAD,
        )
        self.model = GPT2LMHeadModel(config)  # Placeholder - replace with novel architecture

    def forward(self, idx, targets=None):
        outputs = self.model(idx, labels=targets)
        logits = outputs.logits
        loss = outputs.loss if targets is not None else None
        return logits, loss

def run_experiments(tasks, train_baseline=True, train_novel=True):
    for task_name, task in tasks:
        metrics = {}
        if train_baseline:
            print(f"\nTraining Baseline Transformer ({task_name})...")
            baseline_model = BaselineTransformer(task.vocab_size)
            metrics['Baseline'] = task.train_and_evaluate(baseline_model, "Baseline", DEVICE)

        if train_novel:
            print(f"\nTraining Novel Algorithm ({task_name})...")
            novel_model = NovelAlgorithm(task.vocab_size)
            metrics['Novel'] = task.train_and_evaluate(novel_model, "Novel", DEVICE)

        if metrics:
            task.print_comparison(metrics, task_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare language models")
    parser.add_argument('--train_baseline', action='store_true', help="Train the baseline model")
    parser.add_argument('--train_novel', action='store_true', help="Train the novel model")
    parser.add_argument('--tasks', choices=['char', 'token', 'both'], default='both', help="Tasks to run: 'char', 'token', or 'both'")
    args = parser.parse_args()

    train_baseline = args.train_baseline
    train_novel = args.train_novel
    if not (train_baseline or train_novel):
        train_baseline = True
        train_novel = True

    tasks = []
    if args.tasks in ['char', 'both']:
        train_data, val_data, test_data, vocab_size, stoi, itos = get_char_data()
        tasks.append(('Character-Level', CharLanguageModelingTask(train_data, val_data, test_data, vocab_size, stoi, itos)))
    if args.tasks in ['token', 'both']:
        train_data, val_data, test_data, tokenizer = get_token_data()
        tasks.append(('Token-Level', TokenLanguageModelingTask(train_data, val_data, test_data, tokenizer.vocab_size, tokenizer=tokenizer)))

    run_experiments(tasks, train_baseline, train_novel)
