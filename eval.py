import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import math

def train_and_evaluate(model, train_dataloader, eval_dataloader, optimizer, num_epochs, model_name="Model", smoke_test=True):
    """
    A generic training and evaluation function using Hugging Face Accelerate.
    If smoke_test is True, it will only run for one batch.
    """
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} | {model_name} Training", disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(progress_bar):
            if smoke_test and step > 0:
                break
            outputs = model(**batch)
            loss = outputs['loss']
            total_loss += loss.item()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({'loss': loss.item()})

    model.eval()
    losses = []
    eval_progress_bar = tqdm(eval_dataloader, desc=f"{model_name} Evaluation", disable=not accelerator.is_local_main_process)
    for step, batch in enumerate(eval_progress_bar):
        if smoke_test and step > 0:
            break
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs['loss']
        losses.append(accelerator.gather(loss.repeat(batch['input_ids'].shape[0])))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataloader.dataset)]
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    return {
        'eval_loss': eval_loss.item(),
        'perplexity': perplexity
    }
