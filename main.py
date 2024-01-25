import torch
import os

import torch.nn as nn
import numpy as np
import albumentations as A
from loguru import logger

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from albumentations.pytorch import ToTensorV2

from model import CNNLSTM
from dataloader import VideoDataset
from train import train_model
from evaluate import evaluate_model
from utils import load_params, visualize_history


def main():
    params = load_params("params.json")

    hidden_size = params["hidden_size"]
    num_lstm_layers = params["num_lstm_layers"]
    use_pretrained = params["use_pretrained"]
    num_frames = params["num_frames"]
    num_classes = params["num_classes"]
    batch_size = params["batch_size"]
    num_workers = params["num_workers"]
    device = params["device"]

    img_size = params["img_size"]
    data_dir = params["data_dir"]
    epochs = params["epochs"]

    cache_dir = params["cache_dir"]
    last_weights = os.path.join(cache_dir, params["last_weights"])
    best_weights = os.path.join(cache_dir, params["best_weights"])

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    # Create full paths
    train_indices_path = os.path.join(cache_dir, "train_indices.npy")
    val_indices_path = os.path.join(cache_dir, "val_indices.npy")
    test_indices_path = os.path.join(cache_dir, "test_indices.npy")

    transform = A.Compose(
        [A.Resize(height=img_size[0], width=img_size[1]), A.Normalize(), ToTensorV2()]
    )

    logger.info("Loading dataset")
    full_dataset = VideoDataset(
        data_dir=data_dir,
        num_frames=num_frames,
        num_classes=num_classes,
        transform=transform,
    )

    # Check if indices files exist
    if (
        os.path.exists(train_indices_path)
        and os.path.exists(val_indices_path)
        and os.path.exists(test_indices_path)
    ):
        # Load indices
        train_idx = np.load(train_indices_path)
        val_idx = np.load(val_indices_path)
        test_idx = np.load(test_indices_path)
    else:
        # Split the dataset and save indices
        train_idx, temp_idx = train_test_split(
            np.arange(len(full_dataset)), test_size=0.3, random_state=42
        )
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

        # Save indices
        np.save(train_indices_path, train_idx)
        np.save(val_indices_path, val_idx)
        np.save(test_indices_path, test_idx)

    # Create subsets
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    logger.info("Loading model")
    model = CNNLSTM(
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        use_pretrained=use_pretrained,
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    logger.info("Starting training process")

    if params["train_model"]:
        model, history = train_model(
            model,
            train_loader,
            loss_fn,
            optimizer,
            epochs=epochs,
            device=device,
            val_data=val_loader,
            save_best_path=best_weights,
        )
        torch.save(model.state_dict(), last_weights)
        logger.info(f"Saved last model state to: {last_weights}")
        visualize_history(history, save_path="results.png")
    else:
        weights_path = best_weights if params["use_best_model"] else last_weights
        model.load_state_dict(torch.load(weights_path))
        print(f"Loaded model weights from {weights_path}")

    logger.info("Evaluating on test dataset")
    test_loss, test_acc = evaluate_model(
        model,
        weights=last_weights,
        val_data=test_loader,
        loss_fn=loss_fn,
        device=device,
        verbose=1,
    )
    print(f"Loss: {test_loss:.3f}, Acc: {test_acc:.3f}")

    test_loss, test_acc = evaluate_model(
        model,
        weights=best_weights,
        val_data=test_loader,
        loss_fn=loss_fn,
        device=device,
        verbose=1,
    )
    print(f"Loss: {test_loss:.3f}, Acc: {test_acc:.3f}")


if __name__ == "__main__":
    main()
