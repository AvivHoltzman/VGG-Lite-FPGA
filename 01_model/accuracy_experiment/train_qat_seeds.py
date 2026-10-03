#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_qat_seeds.py
==================
Trains the quantized (QAT) VGG-Lite model several times with different random
seeds, to measure how much its accuracy varies from run to run.

PURPOSE
-------
Together with the three FP32 runs of train_fp32_reference.py, these runs give
both sides of the comparison their own mean and spread, so the cost of
quantization is stated as a comparison between two distributions (Welch's
t-test) rather than between two single numbers.

SAFETY
------
This script NEVER touches vgg_lite_int8_best.pth -- the checkpoint the deployed
bitstream corresponds to. Outputs go to vgg_lite_int8_seed<N>.pth. The script
refuses to start if the output path would collide with the deployed checkpoint.

The models trained here are for measuring variance only. The hardware, the
bitstream and the deployed weights stay exactly as they are.

RECIPE
------
Identical to train_vgg_lite_full.py, which is what the deployed model used:
    batch size 64, Adam(lr=0.001), CosineAnnealingLR(T_max=20), 20 epochs,
    RandomCrop(32, padding=4) + RandomHorizontalFlip, same normalization,
    best epoch selected by test accuracy.
The only differences: the random seed is set explicitly, and the script runs
on the GPU when available.

USAGE
-----
    cd /home/avivh/finn
    python3 train_qat_seeds.py --cifar-raw /home/avivh/finn/cifar_data --seeds 42 1 2

Expect roughly 3-6x the FP32 run time per seed -- Brevitas adds fake-quantization
work to every forward and backward pass.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the data pipeline from the FP32 script so both experiments read CIFAR-10
# through exactly the same code. A preprocessing difference between them would
# invalidate the comparison silently.
from train_fp32_reference import (
    RawCIFAR10, find_cifar_dir, set_seed, evaluate, train_one_epoch,
    CIFAR10_MEAN, CIFAR10_STD,
)

DEPLOYED_CHECKPOINT = "vgg_lite_int8_best.pth"


def build_loaders(cifar_raw, batch_size, num_workers):
    print(f"data source   : raw CIFAR-10 pickles at {find_cifar_dir(cifar_raw)}")
    train_set = RawCIFAR10(cifar_raw, train=True, augment=True)
    test_set = RawCIFAR10(cifar_raw, train=False, augment=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=num_workers)
    print(f"train images  : {len(train_set):,}")
    print(f"test images   : {len(test_set):,}")
    return train_loader, test_loader


def run_one_seed(seed, args, device):
    print("\n" + "=" * 70)
    print(f"QAT TRAINING -- SEED {seed}")
    print("=" * 70)

    set_seed(seed)

    from vgg_lite_model import VGG_Lite_INT8
    model = VGG_Lite_INT8().to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device        : {device}")
    print(f"batch size    : {args.batch_size}")
    print(f"epochs        : {args.epochs}")
    print(f"parameters    : {n_params:,}")

    train_loader, test_loader = build_loaders(args.cifar_raw, args.batch_size,
                                              args.num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_path = os.path.join(args.out_dir, f"vgg_lite_int8_seed{seed}.pth")
    if os.path.basename(out_path) == DEPLOYED_CHECKPOINT:
        raise SystemExit("refusing to overwrite the deployed checkpoint")

    best_acc = 0.0
    history = []
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
            torch.save({"state_dict": model.state_dict(),
                        "epoch": epoch,
                        "test_acc": test_acc,
                        "seed": seed}, out_path)

        history.append({"epoch": epoch,
                        "train_loss": round(train_loss, 4),
                        "train_acc": round(train_acc, 2),
                        "test_acc": round(test_acc, 2),
                        "seconds": round(dt, 1)})

        print(f"epoch {epoch:2d}/{args.epochs}  loss {train_loss:.4f}  "
              f"train {train_acc:5.2f}%  test {test_acc:5.2f}% ({correct}/{total})  "
              f"{dt:5.1f}s{'  <-- best' if is_best else ''}")

    minutes = (time.time() - t_start) / 60
    print(f"\nbest QAT test accuracy (seed {seed}) : {best_acc:.2f}%")
    print(f"time                                : {minutes:.1f} min")
    print(f"checkpoint                          : {out_path}")

    return {"seed": seed, "best_test_acc": best_acc,
            "minutes": round(minutes, 1), "checkpoint": out_path,
            "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifar-raw", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--summary", default="qat_seeds_summary.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    results = [run_one_seed(s, args, device) for s in args.seeds]

    accs = [r["best_test_acc"] for r in results]
    mean = sum(accs) / len(accs)
    if len(accs) > 1:
        var = sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)
        std = var ** 0.5
    else:
        std = 0.0

    print("\n" + "=" * 70)
    print("QAT SEED SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  seed {r['seed']:<4} : {r['best_test_acc']:.2f}%")
    print("-" * 70)
    print(f"  mean       : {mean:.2f}%")
    print(f"  std (n-1)  : {std:.2f}")
    print(f"  range      : {min(accs):.2f}% - {max(accs):.2f}%")
    print("=" * 70)
    print()
    print("Compare this against the FP32 seed statistics. If the two ranges")
    print("overlap, the honest statement is that quantization to INT8 cost no")
    print("accuracy that this experiment can resolve.")
    print()
    print("NOTE: the deployed model remains vgg_lite_int8_best.pth. These")
    print("checkpoints exist only to measure run-to-run variance.")

    with open(os.path.join(args.out_dir, args.summary), "w") as f:
        json.dump({"mean": round(mean, 2), "std": round(std, 2),
                   "accuracies": accs, "runs": results}, f, indent=2)
    print(f"\nsummary written to {os.path.join(args.out_dir, args.summary)}")


if __name__ == "__main__":
    main()
