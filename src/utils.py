import logging
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set global random seed for reproducibility across common ML libraries."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_logger(name: str, log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """Create a console + file logger with a standard production-style format."""

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False

    if logger.handlers:
        return logger

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{name}.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy for a classification batch."""

    predictions = torch.argmax(logits, dim=1)
    correct_count = (predictions == targets).sum().item()
    batch_size = targets.size(0)
    return correct_count / batch_size if batch_size > 0 else 0.0


def save_checkpoint(state: Dict, checkpoint_path: str) -> None:
    """Persist model checkpoint to disk."""

    checkpoint_dir = Path(checkpoint_path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)
