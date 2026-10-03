#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_fp32_reference.py
=======================
Trains a plain FP32 reference model with exactly the same architecture and the
same training recipe as the quantized (QAT) VGG-Lite model.

PURPOSE
-------
To measure the accuracy cost of 8-bit quantization, the INT8 model must be
compared against a true floating-point model: same layers, same channel counts,
same kernel sizes, same pooling, same classifier, same optimizer, same schedule
and the same number of epochs -- with the Quant* layers replaced by their
standard torch.nn equivalents. Only the number format differs, so any accuracy
gap is the cost of quantization.

WHERE TO RUN
------------
Run this in the same WSL environment used to train the QAT model (torch,
torchvision and the CIFAR-10 dataset are already there). It auto-detects CUDA
and falls back to CPU.

    python3 train_fp32_reference.py --data-root ./data --epochs 20

Expected runtime: a few minutes on a GPU, roughly 1-2 hours on a desktop CPU.

DO NOT run this on the PYNQ board's ARM CPU. The accuracy of a trained model
does not depend on which processor trained it, and the Cortex-A9 would need
tens of hours (PyTorch has no armv7 wheels in any case). The ARM is needed only
for measuring INFERENCE TIME.

RECIPE (identical to train_vgg_lite_full.py)
--------------------------------------------
    batch size 64, 20 epochs, Adam(lr=0.001), CosineAnnealingLR(T_max=epochs),
    RandomCrop(32, padding=4) + RandomHorizontalFlip, the same normalization.
The reported runs (seeds 42, 1, 2) used --batch-size 64 --epochs 20; their
exact arguments are stored in 05_results/fp32_log_seed*.json.
"""

import argparse
import json
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# torchvision is optional. Inside the FINN Docker container it is often absent
# or unable to download CIFAR-10, so everything it provides has a fallback
# below. Use --cifar-raw to point at the extracted cifar-10-batches-py folder.
try:
    from torchvision import datasets, transforms
    HAVE_TORCHVISION = True
except ImportError:
    datasets = None
    transforms = None
    HAVE_TORCHVISION = False

# ---------------------------------------------------------------------------
# Recipe constants -- must mirror train_vgg_lite_full.py
# ---------------------------------------------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 0.001


# ---------------------------------------------------------------------------
# Model: VGG-Lite V3, FP32 version
# ---------------------------------------------------------------------------
class VGGLiteFP32(nn.Module):
    """FP32 twin of the quantized VGG-Lite V3.

    Layer-by-layer correspondence with the Brevitas model:

        QuantConv2d(3, 16, 3, pad=1)  -> nn.Conv2d(3, 16, 3, padding=1)
        QuantReLU                     -> nn.ReLU
        QuantConv2d(16, 16, 3, pad=1) -> nn.Conv2d(16, 16, 3, padding=1)
        QuantReLU                     -> nn.ReLU
        MaxPool2d(2, 2)               -> MaxPool2d(2, 2)          [identical]
        QuantConv2d(16, 32, 3, pad=1) -> nn.Conv2d(16, 32, 3, padding=1)
        QuantReLU                     -> nn.ReLU
        QuantConv2d(32, 32, 3, pad=1) -> nn.Conv2d(32, 32, 3, padding=1)
        QuantReLU                     -> nn.ReLU
        MaxPool2d(2, 2)               -> MaxPool2d(2, 2)          [identical]
        AvgPool2d(4, 4)               -> AvgPool2d(4, 4)          [identical]
        Flatten                       -> Flatten                  [identical]
        QuantLinear(128, 10, bias=True) -> nn.Linear(128, 10, bias=True)

    Tensor shapes, for the record:
        input            (N, 3, 32, 32)
        after block 1    (N, 16, 16, 16)
        after block 2    (N, 32,  8,  8)
        after AvgPool    (N, 32,  2,  2)  -> flatten -> 128
        output           (N, 10)
    """

    def __init__(self, num_classes=10, conv_bias=False):
        super().__init__()

        self.features = nn.Sequential(
            # --- block 1 ---
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=conv_bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=conv_bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # --- block 2 ---
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=conv_bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=conv_bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Linear(128, num_classes, bias=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed):
    """Make the run reproducible so the reported number can be regenerated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# CIFAR-10 without torchvision
# ---------------------------------------------------------------------------
def find_cifar_dir(root):
    """Accept either the cifar-10-batches-py folder itself or its parent."""
    candidates = [root, os.path.join(root, "cifar-10-batches-py")]
    for c in candidates:
        if os.path.exists(os.path.join(c, "test_batch")):
            return c
    raise FileNotFoundError(
        f"Could not find 'test_batch' under {root} or "
        f"{os.path.join(root, 'cifar-10-batches-py')}.\n"
        f"Download cifar-10-python.tar.gz, extract it, and point --cifar-raw "
        f"at the resulting cifar-10-batches-py folder.")


class RawCIFAR10(Dataset):
    """CIFAR-10 read straight from the official python pickle batches.

    The test split is test_batch in its original order -- the same order that
    extract_reference.py used -- so image i here is image i there. That matters:
    it lets hardware predictions be compared image-for-image with these labels.
    """

    def __init__(self, root, train, augment):
        d = find_cifar_dir(root)
        files = ([f"data_batch_{i}" for i in range(1, 6)] if train
                 else ["test_batch"])

        chunks = []
        labels = []
        for fn in files:
            path = os.path.join(d, fn)
            with open(path, "rb") as f:
                entry = pickle.load(f, encoding="bytes")
            chunks.append(entry[b"data"])
            labels.extend(entry[b"labels"])

        # (N, 3072) uint8, stored as R-plane then G-plane then B-plane
        raw = np.concatenate(chunks, axis=0)
        self.images = raw.reshape(-1, 3, 32, 32).astype(np.uint8)  # NCHW
        self.labels = np.asarray(labels, dtype=np.int64)
        self.augment = augment

        self.mean = np.asarray(CIFAR10_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(CIFAR10_STD, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]

        if self.augment:
            # RandomCrop(32, padding=4): zero-pad by 4 on every side, then take
            # a random 32x32 window. Matches torchvision's default fill=0.
            padded = np.zeros((3, 40, 40), dtype=np.uint8)
            padded[:, 4:36, 4:36] = img
            top = np.random.randint(0, 9)
            left = np.random.randint(0, 9)
            img = padded[:, top:top + 32, left:left + 32]

            # RandomHorizontalFlip(p=0.5)
            if np.random.rand() < 0.5:
                img = img[:, :, ::-1]

        # ToTensor(): uint8 [0,255] -> float32 [0,1]; then Normalize()
        x = np.ascontiguousarray(img).astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        return torch.from_numpy(x), int(self.labels[idx])


def build_loaders(data_root, batch_size, num_workers, cifar_raw=None):
    """Return (train_loader, test_loader).

    If --cifar-raw is given, or torchvision is unavailable, read the pickles
    directly. Both paths produce identical tensors for the test split.
    """
    use_raw = cifar_raw is not None or not HAVE_TORCHVISION

    if use_raw:
        root = cifar_raw if cifar_raw else data_root
        print(f"data source   : raw CIFAR-10 pickles at {find_cifar_dir(root)}")
        train_set = RawCIFAR10(root, train=True, augment=True)
        test_set = RawCIFAR10(root, train=False, augment=False)
    else:
        print(f"data source   : torchvision.datasets.CIFAR10 at {data_root}")
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        train_set = datasets.CIFAR10(root=data_root, train=True,
                                     download=True, transform=train_tf)
        test_set = datasets.CIFAR10(root=data_root, train=False,
                                    download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=num_workers)
    print(f"train images  : {len(train_set):,}")
    print(f"test images   : {len(test_set):,}")
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    """Top-1 accuracy over the whole loader. Returns (accuracy_pct, n_correct, n_total)."""
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / total, correct, total


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.numel()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.numel()

    return running_loss / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train the FP32 reference model for the VGG-Lite accelerator project.")
    parser.add_argument("--data-root", default="./data",
                        help="CIFAR-10 root for torchvision (downloads if absent)")
    parser.add_argument("--cifar-raw", default=None,
                        help="Path to the extracted cifar-10-batches-py folder. "
                             "Use this inside the FINN Docker container, where "
                             "torchvision usually cannot download the dataset.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--conv-bias", action="store_true",
                        help="Enable bias on conv layers (set this only if the "
                             "Brevitas model used bias=True on its QuantConv2d layers)")
    parser.add_argument("--out", default="vgg_lite_fp32_best.pth")
    parser.add_argument("--log", default="fp32_training_log.json")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu. Default: cuda when available.")
    args = parser.parse_args()

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    set_seed(args.seed)

    print("=" * 70)
    print("FP32 REFERENCE TRAINING -- VGG-Lite V3")
    print("=" * 70)
    print(f"device        : {device}")
    if device.type == "cuda":
        print(f"gpu           : {torch.cuda.get_device_name(0)}")
    print(f"epochs        : {args.epochs}")
    print(f"batch size    : {args.batch_size}")
    print(f"optimizer     : Adam(lr={args.lr})")
    print(f"lr schedule   : CosineAnnealingLR(T_max={args.epochs})")
    print(f"conv bias     : {args.conv_bias}")
    print(f"seed          : {args.seed}")
    print("=" * 70)

    train_loader, test_loader = build_loaders(args.data_root, args.batch_size,
                                              args.num_workers, args.cifar_raw)

    model = VGGLiteFP32(num_classes=10, conv_bias=args.conv_bias).to(device)
    print(f"trainable parameters: {count_parameters(model):,}")
    print("(the quantized model has the same count -- quantization changes the")
    print(" width of each value, not how many values there are)")
    print("-" * 70)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_acc = 0.0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                optimizer, device)
        test_acc, correct, total = evaluate(model, test_loader, device)
        scheduler.step()
        dt = time.time() - t0

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "test_acc": test_acc,
                "conv_bias": args.conv_bias,
                "recipe": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "optimizer": "Adam",
                    "scheduler": "CosineAnnealingLR",
                    "seed": args.seed,
                    "normalize_mean": CIFAR10_MEAN,
                    "normalize_std": CIFAR10_STD,
                },
            }, args.out)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "test_acc": round(test_acc, 2),
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(dt, 1),
        })

        marker = "  <-- best" if is_best else ""
        print(f"epoch {epoch:2d}/{args.epochs}  "
              f"loss {train_loss:.4f}  train {train_acc:5.2f}%  "
              f"test {test_acc:5.2f}% ({correct}/{total})  "
              f"{dt:5.1f}s{marker}")

    total_time = time.time() - t_start

    print("-" * 70)
    print(f"best FP32 test accuracy : {best_acc:.2f}%  (full 10,000-image test set)")
    print(f"total training time     : {total_time/60:.1f} min")
    print(f"checkpoint saved to     : {args.out}")
    print("=" * 70)
    print()
    print("NEXT STEP: run compare_models.py to evaluate the FP32 model and the")
    print("QAT model on the full test set with identical evaluation code, so the")
    print("two accuracy numbers in the project book come from the same measurement.")

    with open(args.log, "w") as f:
        json.dump({
            "best_test_acc": best_acc,
            "total_minutes": round(total_time / 60, 1),
            "device": str(device),
            "args": vars(args),
            "history": history,
        }, f, indent=2)
    print(f"training log saved to   : {args.log}")


if __name__ == "__main__":
    main()
