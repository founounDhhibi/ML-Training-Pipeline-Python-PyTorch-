import logging
from typing import Dict

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.utils import accuracy_from_logits


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    logger: logging.Logger,
    log_every_n_steps: int = 100,
) -> Dict[str, float]:
    """Run one training epoch and return aggregate metrics."""

    model.train()

    epoch_loss_sum = 0.0
    epoch_correct_predictions = 0
    epoch_examples = 0

    for step_index, (features, labels) in enumerate(data_loader, start=1):
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        batch_accuracy = accuracy_from_logits(logits, labels)

        epoch_loss_sum += loss.item() * batch_size
        epoch_correct_predictions += int(batch_accuracy * batch_size)
        epoch_examples += batch_size

        if step_index % log_every_n_steps == 0:
            logger.info(
                "train_step=%s loss=%.4f batch_acc=%.4f",
                step_index,
                loss.item(),
                batch_accuracy,
            )

    epoch_loss = epoch_loss_sum / epoch_examples
    epoch_accuracy = epoch_correct_predictions / epoch_examples

    return {"loss": epoch_loss, "accuracy": epoch_accuracy}
