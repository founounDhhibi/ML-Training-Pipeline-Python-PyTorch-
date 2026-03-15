from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils import accuracy_from_logits


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model and return aggregate loss + accuracy."""

    model.eval()

    loss_sum = 0.0
    correct_predictions = 0
    total_examples = 0

    for features, labels in data_loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        batch_accuracy = accuracy_from_logits(logits, labels)

        loss_sum += loss.item() * batch_size
        correct_predictions += int(batch_accuracy * batch_size)
        total_examples += batch_size

    average_loss = loss_sum / total_examples
    average_accuracy = correct_predictions / total_examples

    return {"loss": average_loss, "accuracy": average_accuracy}
