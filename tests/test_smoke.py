import torch

from src.model import SimpleCNN
from src.utils import accuracy_from_logits, set_seed


def test_model_forward_shape() -> None:
    set_seed(123)
    model = SimpleCNN(num_classes=10)
    features = torch.randn(4, 1, 28, 28)

    logits = model(features)

    assert logits.shape == (4, 10)


def test_accuracy_helper() -> None:
    logits = torch.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
        ]
    )
    labels = torch.tensor([0, 2, 1])

    accuracy = accuracy_from_logits(logits, labels)

    assert accuracy == 2 / 3
