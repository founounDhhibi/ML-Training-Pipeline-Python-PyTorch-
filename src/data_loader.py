from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_data_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 2,
    validation_split: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/validation/test dataloaders for MNIST."""

    # Standard MNIST preprocessing.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )

    train_val_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=True,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=True,
    )

    val_size = int(len(train_val_dataset) * validation_split)
    train_size = len(train_val_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        train_val_dataset,
        lengths=[train_size, val_size],
        generator=generator,
    )

    # Pin memory improves host-to-device transfer for CUDA workloads.
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
