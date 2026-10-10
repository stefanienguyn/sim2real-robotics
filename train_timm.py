import argparse
import csv
import json
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import timm

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="JetBot line-following trainer")

    # Data
    p.add_argument("--data-dir", required=True, help="Root dataset directory")
    p.add_argument("--output-dir", default="output/line_following",
                   help="Where to save runs (a timestamped subdir is created)")
    p.add_argument("--val-split", type=float, default=0.15,
                   help="Fraction of data to use for validation when no train/val split exists")

    # Model
    p.add_argument("--model", default="mobilenetv3_small_050",
                   help="TIMM model name (e.g. mobilenetv3_small_050, resnet18)")
    p.add_argument("--pretrained", action="store_true",
                   help="Load ImageNet pretrained weights")
    p.add_argument("--img-size", type=int, default=224)

    # Training
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="", help="cuda / cpu (auto-detected if empty)")
    p.add_argument("--patience", type=int, default=0,
                   help="Early-stopping patience in epochs (0 = disabled)")
    p.add_argument("--resume", default="", help="Path to a checkpoint to resume from")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),   # slight oversize then center-crop
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def has_split(data_dir: Path) -> bool:
    """Return True if data_dir already has train/ and val/ subdirectories."""
    return (data_dir / "train").is_dir() and (data_dir / "val").is_dir()


def make_datasets(args):
    data_dir = Path(args.data_dir)
    train_tf, val_tf = build_transforms(args.img_size)

    if has_split(data_dir):
        print("Found explicit train/val split.")
        train_ds = datasets.ImageFolder(str(data_dir / "train"), transform=train_tf)
        val_ds   = datasets.ImageFolder(str(data_dir / "val"),   transform=val_tf)
    else:
        print(f"No train/val split found — using {args.val_split:.0%} of data for validation.")
        full_ds = datasets.ImageFolder(str(data_dir))

        # Reproducible split
        rng = random.Random(args.seed)
        indices = list(range(len(full_ds)))
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * args.val_split))
        val_idx   = indices[:n_val]
        train_idx = indices[n_val:]

        # Wrap with per-split transforms via a small helper
        train_ds = TransformSubset(full_ds, train_idx, train_tf)
        val_ds   = TransformSubset(full_ds, val_idx,   val_tf)

    # Verify exactly 3 classes
    expected = {"left", "forward", "right"}
    found = set(train_ds.classes if hasattr(train_ds, "classes")
                else train_ds.dataset.classes)
    missing = expected - found
    if missing:
        raise ValueError(
            f"Dataset is missing classes: {missing}. "
            f"Found: {found}. Expected exactly: {expected}"
        )

    return train_ds, val_ds


class TransformSubset(Subset):
    """A Subset that applies its own transform, overriding the parent dataset's."""

    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def class_to_idx(self):
        return self.dataset.class_to_idx

    def __getitem__(self, idx):
        img, label = self.dataset.imgs[self.indices[idx]]
        from PIL import Image
        img = Image.open(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def accuracy(outputs, targets):
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()


def update_confusion(matrix, outputs, targets, n_classes):
    preds = outputs.argmax(dim=1).cpu()
    targets = targets.cpu()
    for t, p in zip(targets, preds):
        matrix[t.item()][p.item()] += 1


def print_confusion(matrix, classes):
    print("\n  Confusion matrix (rows=true, cols=pred):")
    header = "       " + "  ".join(f"{c:>8}" for c in classes)
    print(header)
    for i, row in enumerate(matrix):
        row_str = "  ".join(f"{v:>8}" for v in row)
        print(f"  {classes[i]:>6} {row_str}")
    print()


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n


@torch.no_grad()
def validate(model, loader, criterion, device, n_classes):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    n = 0
    confusion = [[0] * n_classes for _ in range(n_classes)]

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        total_correct += (out.argmax(1) == labels).sum().item()
        update_confusion(confusion, out, labels, n_classes)
        n += imgs.size(0)

    return total_loss / n, total_correct / n, confusion


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, class_to_idx, args, val_acc, epoch):
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_to_idx":     class_to_idx,
        "idx_to_class":     idx_to_class,
        "model_name":       args.model,
        "img_size":         args.img_size,
        "val_acc":          val_acc,
        "epoch":            epoch,
    }, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Save args
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Datasets & loaders
    train_ds, val_ds = make_datasets(args)
    class_to_idx = (train_ds.class_to_idx
                    if hasattr(train_ds, "class_to_idx")
                    else train_ds.dataset.class_to_idx)
    classes = sorted(class_to_idx, key=class_to_idx.get)
    n_classes = len(classes)
    print(f"Classes: {classes}  (mapping: {class_to_idx})")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Save class mapping
    with open(run_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    model = timm.create_model(
        args.model, pretrained=args.pretrained, num_classes=n_classes
    )
    model = model.to(device)
    print(f"Model: {args.model}  |  pretrained={args.pretrained}")

    # Optionally resume
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("val_acc", 0.0)
        print(f"Resumed from {args.resume}  (epoch {start_epoch}, best_acc {best_val_acc:.4f})")

    # Optimizer & loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()

    # CSV log
    log_path = run_dir / "training_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "lr", "elapsed_s"])

    best_epoch = start_epoch
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, confusion = validate(
            model, val_loader, criterion, device, n_classes
        )
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1:>3}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"lr={current_lr:.2e}  "
            f"({elapsed:.1f}s)"
        )
        print_confusion(confusion, classes)

        log_writer.writerow([epoch+1, f"{train_loss:.6f}", f"{val_loss:.6f}",
                              f"{val_acc:.6f}", f"{current_lr:.2e}", f"{elapsed:.1f}"])
        log_file.flush()

        # Save last
        save_checkpoint(run_dir / "last_model.pth", model, class_to_idx, args, val_acc, epoch)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(run_dir / "best_model.pth", model, class_to_idx, args, val_acc, epoch)
            print(f"  ✓ New best model saved (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs without improvement.")
            break

    log_file.close()
    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f} at epoch {best_epoch+1}.")
    print(f"Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()