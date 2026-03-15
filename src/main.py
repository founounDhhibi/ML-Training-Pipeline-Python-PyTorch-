import argparse
from pathlib import Path

import torch
from torch import nn

from src.data_loader import get_data_loaders
from src.evaluate import evaluate
from src.model import SimpleCNN
from src.train import train_one_epoch
from src.utils import get_logger, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production-style MNIST training pipeline")
    parser.add_argument("--data-dir", type=str, default="data", help="Dataset cache directory")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output checkpoint directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--log-level", type=str, default="INFO", help="Python logging level")
    return parser.parse_args()


def run_training() -> None:
    args = parse_args()

    set_seed(args.seed)
    logger = get_logger(name="ml_pipeline", log_level=args.log_level)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)
    logger.info("seed=%d", args.seed)

    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_accuracy = 0.0
    best_checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            logger=logger,
        )
        val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        logger.info(
            "epoch=%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["accuracy"],
            val_metrics["loss"],
            val_metrics["accuracy"],
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": best_val_accuracy,
                },
                str(best_checkpoint_path),
            )
            logger.info("saved_best_checkpoint=%s", best_checkpoint_path)

    test_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )
    logger.info(
        "test_loss=%.4f test_acc=%.4f",
        test_metrics["loss"],
        test_metrics["accuracy"],
    )


if __name__ == "__main__":
    run_training()
