import torch
import torch as th
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as T
from torch.utils.data import DataLoader
import yaml
import os
import argparse
import logging
from tqdm import tqdm
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional

from model import YOLOv1
from dataset import PigDataset
from loss import YOLO_Loss
import utils


def set_seed(seed: Optional[int]):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Seed: {seed}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def target_to_grid(targets: List[th.Tensor], S: int, C: int) -> th.Tensor:
    batch_size = len(targets)
    target_grid = torch.zeros((batch_size, S, S, C + 5), dtype=torch.float32)
    for batch_idx, target_img in enumerate(targets):
        if target_img.numel() == 0:
            continue

        for obj_idx in range(target_img.shape[0]):
            class_id, cx, cy, w, h = target_img[obj_idx]
            class_id = int(class_id)
            j_cell, i_cell = int(S * cx), int(S * cy)
            x_cell, y_cell = S * cx - j_cell, S * cy - i_cell
            if target_grid[batch_idx, i_cell, j_cell, 0] == 0:
                target_grid[batch_idx, i_cell, j_cell, 0] = 1
                if 0 <= class_id < C:
                    target_grid[batch_idx, i_cell, j_cell, 1 + class_id] = 1
                else:
                    logging.warning(f"Wrong class_id {class_id}. Skipped.")
                    target_grid[batch_idx, i_cell, j_cell, 0] = 0
                    continue
                target_grid[batch_idx, i_cell, j_cell, C + 1] = x_cell
                target_grid[batch_idx, i_cell, j_cell, C + 2] = y_cell
                target_grid[batch_idx, i_cell, j_cell, C + 3] = w
                target_grid[batch_idx, i_cell, j_cell, C + 4] = h

    return target_grid


class YoloCollateFn:
    def __init__(self, S: int, C: int):
        self.S = S
        self.C = C

    def __call__(
        self, batch: List[Tuple[th.Tensor, th.Tensor]]
    ) -> Tuple[th.Tensor, th.Tensor]:
        images = torch.stack([item[0] for item in batch], dim=0)
        targets = [item[1] for item in batch]
        targets_grid_batch = target_to_grid(targets, self.S, self.C)
        return images, targets_grid_batch


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    save_dir: str,
    last_filename: str = "last_checkpoint.pth.tar",
    best_filename_prefix: str = "best_checkpoint",
):
    os.makedirs(save_dir, exist_ok=True)
    last_filepath = os.path.join(save_dir, last_filename)
    torch.save(state, last_filepath)
    logging.info(f"Saved last checkpoint: {last_filepath}")
    if is_best:
        best_filename = f"{best_filename_prefix}_epoch{state['epoch']}_map{state['best_map']:.4f}.pth.tar"
        best_filepath = os.path.join(save_dir, best_filename)
        torch.save(state, best_filepath)
        logging.info(f"Saved best ckpt: {best_filepath} (mAP: {state['best_map']:.4f})")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Optional[Dict[str, Any]]:
    if not os.path.exists(filepath):
        logging.warning(f"File checkpoint not found: {filepath}")
        return None

    try:
        map_location = torch.device(device)
        checkpoint = torch.load(filepath, map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logging.info(f"Checkpoint successfuly loaded from {filepath}")
        return checkpoint

    except Exception as e:
        logging.error(f"Error reading ckpt: {filepath}: {e}")
        return None


def get_optimizer(
    model: YOLOv1, config: Dict[str, Any], backbone_frozen: bool
) -> optim.Optimizer:
    lr_head = float(config["training"]["learning_rate"])
    lr_backbone = float(config["training"]["lr_fine_tune"])
    weight_decay = float(config["training"]["weight_decay"])

    if config["model"]["backbone"] == "custom" or backbone_frozen:
        parameters = model.parameters()
        current_lr = lr_head
        logging.info(
            f"Optimizer: Adam, LR={current_lr:.1e}, WD={weight_decay} (All params)"
        )
    else:
        parameters = [
            {"params": model.backbone.parameters(), "lr": lr_backbone},
            {"params": model.detection_head.parameters(), "lr": lr_head},
        ]
        current_lr = f"Backbone LR={lr_backbone:.1e}, Head LR={lr_head:.1e}"
        logging.info(f"Optimizer: Adam, {current_lr}, WD={weight_decay} (Splitted LR)")

    optimizer = optim.Adam(parameters, lr=lr_head, weight_decay=weight_decay)
    if config["model"]["backbone"] != "custom" and not backbone_frozen:
        optimizer.param_groups[0]["lr"] = lr_backbone
        optimizer.param_groups[1]["lr"] = lr_head
    return optimizer


def train_one_epoch(
    loader: DataLoader,
    model: YOLOv1,
    loss_fn: YOLO_Loss,
    optimizer: optim.Optimizer,
    device: str,
) -> float:
    model.train()
    loop = tqdm(loader, leave=True, desc="Train Epoch")
    total_loss = 0.0
    num_batches = len(loader)

    for batch_idx, (images, targets_grid) in enumerate(loop):
        images, targets_grid = images.to(device), targets_grid.to(device)
        predictions = model(images)
        loss = loss_fn(predictions, targets_grid)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def plot_metrics(
    train_loss_history: List[float], map_history: List[float], save_path: str
):
    if not train_loss_history or not map_history:
        logging.warning("No data for plots")
        return

    epochs = range(1, len(train_loss_history) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color=color)
    ax1.plot(epochs, train_loss_history, color=color, marker="o", label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Validation mAP@0.5 (%)", color=color)
    ax2.plot(
        epochs, map_history, color=color, marker="s", linestyle="--", label="mAP@0.5"
    )
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, max(100, max(map_history) * 1.1) if map_history else 100)

    fig.suptitle("Metrics")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    try:
        plt.savefig(save_path)
        logging.info(f"Saved graphs: {save_path}")
    except Exception as e:
        logging.error(f"Error saving graphs: {e}")
    plt.close(fig)


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    save_dir = config["checkpoint"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "training.log")
    plot_save_path = os.path.join(save_dir, "metrics_plot.png")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info("--- START TRAINING ---")
    logging.info(f"Config loaded from: {config_path}")
    logging.info(f"Config: {config}")

    seed = config.get("training", {}).get("seed", None)
    set_seed(seed)

    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    device = config["training"]["device"] if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    logging.info("Loading data...")
    img_size = config["dataset"]["image_size"]
    S = config["model"]["S"]
    C = config["model"]["C"]
    B = config["model"]["B"]
    normalize = T.Normalize(
        mean=config["dataset"]["normalize_mean"], std=config["dataset"]["normalize_std"]
    )
    train_transforms = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalize,
        ]
    )
    test_transforms = T.Compose(
        [T.Resize((img_size, img_size)), T.ToTensor(), normalize]
    )

    class ApplyTransformsToImage:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, data):
            img, target = data
            img = self.transforms(img)
            return img, target

    try:
        train_dataset = PigDataset(
            img_dir=config["dataset"]["img_dir_train"],
            annot_dir=config["dataset"]["label_dir_train"],
            transforms=ApplyTransformsToImage(train_transforms),
        )
        test_dataset = PigDataset(
            img_dir=config["dataset"]["img_dir_test"],
            annot_dir=config["dataset"]["label_dir_test"],
            transforms=ApplyTransformsToImage(test_transforms),
        )
    except FileNotFoundError as e:
        logging.error(f"Error while loading dataset: {e}")
        return
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        logging.error("Dataset is empty")
        return
    logging.info(f"Training dataset size: {len(train_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")

    yolo_collate_fn = YoloCollateFn(S=S, C=C)

    loader_kwargs = {
        "num_workers": config["training"]["num_workers"],
        "pin_memory": config["training"]["pin_memory"],
    }
    if seed is not None:
        loader_kwargs["worker_init_fn"] = seed_worker
        loader_kwargs["generator"] = g

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=yolo_collate_fn,
        drop_last=True,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False, **loader_kwargs
    )

    logging.info("Init model...")
    model = YOLOv1(S=S, B=B, C=C, backbone_type=config["model"]["backbone"]).to(device)

    backbone_frozen = False
    if (
        config["model"]["backbone"] != "custom"
        and config["training"]["freeze_backbone"]
    ):
        logging.info("Freezing ResNet backbone...")
        for param in model.backbone.parameters():
            param.requires_grad = False
        backbone_frozen = True
    else:
        logging.info("Backbone doesn't freeze.")

    loss_fn = YOLO_Loss(S=S, B=B, C=C, L_coord=5.0, L_noobj=0.5)
    optimizer = get_optimizer(model, config, backbone_frozen)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=5
    )

    start_epoch = 0
    best_map = 0.0
    train_loss_history = []
    map_history = []
    checkpoint_path = os.path.join(save_dir, config["checkpoint"]["save_last_filename"])
    checkpoint_data = load_checkpoint(
        checkpoint_path, model, optimizer, scheduler, device
    )
    if checkpoint_data:
        start_epoch = checkpoint_data.get("epoch", 0)
        best_map = checkpoint_data.get("best_map", 0.0)
        train_loss_history = checkpoint_data.get("train_loss_history", [])
        map_history = checkpoint_data.get("map_history", [])
        if (
            config["model"]["backbone"] != "custom"
            and config["training"]["freeze_backbone"]
            and start_epoch >= config["training"]["unfreeze_epoch"]
        ):
            logging.info(f"Starting from {start_epoch} epoch, unfreezing backbone...")
            for param in model.backbone.parameters():
                param.requires_grad = True
            backbone_frozen = False
            optimizer = get_optimizer(model, config, backbone_frozen)
            if "optimizer_state_dict" in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        logging.info(f"Resuming from {start_epoch + 1} epoch. Best mAP: {best_map:.4f}")

    num_epochs = config["training"]["epochs"]
    logging.info(f"Starting training, {num_epochs} epochs total.")

    for epoch in range(start_epoch, num_epochs):
        current_epoch = epoch + 1
        logging.info(f"\n--- Epoch {current_epoch}/{num_epochs} ---")

        if (
            config["model"]["backbone"] != "custom"
            and backbone_frozen
            and current_epoch == config["training"]["unfreeze_epoch"]
        ):
            logging.info(f"*** Unfreezing ResNet backbone on {current_epoch} epoch ***")
            for param in model.backbone.parameters():
                param.requires_grad = True
            backbone_frozen = False
            old_optimizer_state = optimizer.state_dict()
            optimizer = get_optimizer(model, config, backbone_frozen)
            try:
                optimizer.load_state_dict(old_optimizer_state)
            except ValueError as e:
                pass
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.1, patience=5
            )

        start_time = time.time()
        avg_train_loss = train_one_epoch(
            train_loader, model, loss_fn, optimizer, device
        )
        epoch_duration = time.time() - start_time
        logging.info(
            f"Epoch {current_epoch}, Training is done. Mean loss: {avg_train_loss:.4f}. Time: {epoch_duration:.2f} sec."
        )
        train_loss_history.append(avg_train_loss)

        logging.info(f"Epoch {current_epoch}, validation...")
        start_time = time.time()
        current_map, class_aps = utils.evaluate_model(
            model, test_loader, config, device
        )
        val_duration = time.time() - start_time
        logging.info(
            f"Epoch {current_epoch}, val end. mAP: {current_map:.4f}. Time: {val_duration:.2f} sec."
        )
        if class_aps:
            for cls_aps in class_aps:
                logging.info(f"AP for class 0: {class_aps[0]:.4f}")
        map_history.append(current_map)

        scheduler.step(current_map)

        is_best = current_map > best_map
        if is_best:
            best_map = current_map
            logging.info(f"*** New best mAP: {best_map:.4f} ***")

        save_checkpoint(
            state={
                "epoch": current_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_map": best_map,
                "train_loss_history": train_loss_history,
                "map_history": map_history,
                "config": config,
            },
            is_best=is_best,
            save_dir=save_dir,
            last_filename=config["checkpoint"]["save_last_filename"],
            best_filename_prefix=config["checkpoint"]["save_best_filename_prefix"],
        )

        plot_metrics(train_loss_history, map_history, plot_save_path)

    logging.info("--- TRAINING END ---")
    logging.info(f"best mAP: {best_map:.4f}")
    logging.info(f"final graph saved to: {plot_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training YOLOv1.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="YAML PATH",
    )
    args = parser.parse_args()

    main(args.config)
