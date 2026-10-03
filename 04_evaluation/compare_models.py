#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_models.py
=================
Evaluates the FP32 reference model and the quantized (QAT) model on the FULL
CIFAR-10 test set (10,000 images) using ONE identical evaluation path, and
prints a single table for the project book. Both models go through exactly the
same data loading and evaluation code, so their accuracies are directly
comparable.

USAGE
-----
    python3 compare_models.py \
        --fp32-ckpt vgg_lite_fp32_best.pth \
        --qat-ckpt  vgg_lite_int8_best.pth \
        --qat-module vgg_lite_model \
        --data-root ./data

If --qat-class is omitted the script lists the nn.Module subclasses it found in
the module so you can pick the right one.

WHAT IT REPORTS
---------------
  * top-1 accuracy of each model on all 10,000 test images
  * a 95% confidence interval for each accuracy (so the book can say whether a
    difference is real or within sampling noise)
  * prediction agreement between the two models
  * per-class accuracy for both
  * McNemar test on the disagreements -- the standard test for "are these two
    classifiers actually different?", which is exactly the question an examiner
    will ask about a 1-2% gap

Everything is written to comparison_results.json as well.
"""

import argparse
import importlib
import inspect
import json
import math
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# torchvision is optional -- see train_fp32_reference.py for why.
try:
    from torchvision import datasets, transforms
    HAVE_TORCHVISION = True
except ImportError:
    datasets = None
    transforms = None
    HAVE_TORCHVISION = False

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_test_loader(data_root, cifar_raw=None, batch_size=256, num_workers=2):
    """Test split only, in its original test_batch order.

    Order matters: image i here must be image i in extract_reference.py and on
    the board, so hardware predictions can be compared image-for-image.
    """
    if cifar_raw is not None or not HAVE_TORCHVISION:
        # Reuse the loader from the training script so both scripts read the
        # data identically -- no chance of a preprocessing mismatch.
        from train_fp32_reference import RawCIFAR10, find_cifar_dir
        root = cifar_raw if cifar_raw else data_root
        print(f"data source: raw CIFAR-10 pickles at {find_cifar_dir(root)}")
        test_set = RawCIFAR10(root, train=False, augment=False)
    else:
        print(f"data source: torchvision.datasets.CIFAR10 at {data_root}")
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        test_set = datasets.CIFAR10(root=data_root, train=False,
                                    download=True, transform=test_tf)

    print(f"test images: {len(test_set):,}")
    return DataLoader(test_set, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_fp32(ckpt_path, device):
    """Load the FP32 reference produced by train_fp32_reference.py."""
    from train_fp32_reference import VGGLiteFP32

    ckpt = torch.load(ckpt_path, map_location="cpu")
    conv_bias = ckpt.get("conv_bias", False)
    model = VGGLiteFP32(num_classes=10, conv_bias=conv_bias)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    meta = {
        "checkpoint": ckpt_path,
        "epoch": ckpt.get("epoch"),
        "test_acc_at_save": ckpt.get("test_acc"),
        "conv_bias": conv_bias,
    }
    return model, meta


def list_module_classes(module):
    """Return the nn.Module subclasses defined in a module (for error messages)."""
    found = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, nn.Module) and obj.__module__ == module.__name__:
            found.append(name)
    return found


def load_qat(module_name, class_name, ckpt_path, device):
    """Load the Brevitas QAT model from the project's own model definition."""
    sys.path.insert(0, ".")
    try:
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"\nERROR: could not import '{module_name}'.\n"
            f"       Run this script from the directory that contains "
            f"{module_name}.py, or pass --qat-module with the right name.\n"
            f"       ({exc})\n")

    available = list_module_classes(mod)
    if class_name is None:
        raise SystemExit(
            f"\nERROR: --qat-class not given.\n"
            f"       nn.Module classes found in {module_name}.py:\n"
            + "".join(f"         --qat-class {c}\n" for c in available))

    if not hasattr(mod, class_name):
        raise SystemExit(
            f"\nERROR: class '{class_name}' not found in {module_name}.py.\n"
            f"       Available: {', '.join(available) or '(none)'}\n")

    model = getattr(mod, class_name)()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    # The QAT checkpoint may be a bare state_dict or a wrapped dict.
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  note: {len(missing)} keys missing from checkpoint "
              f"(first few: {missing[:3]})")
    if unexpected:
        print(f"  note: {len(unexpected)} unexpected keys in checkpoint "
              f"(first few: {unexpected[:3]})")

    model.to(device).eval()
    return model, {"checkpoint": ckpt_path,
                   "module": module_name,
                   "class": class_name}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run the model over the whole loader; return (predictions, labels) arrays."""
    preds_all = []
    labels_all = []
    for images, labels in loader:
        images = images.to(device)
        out = model(images)
        # Brevitas QuantTensor -> take .value if present
        if hasattr(out, "value"):
            out = out.value
        preds_all.append(out.argmax(dim=1).cpu().numpy())
        labels_all.append(labels.numpy())
    return np.concatenate(preds_all), np.concatenate(labels_all)


def wilson_interval(correct, total, z=1.96):
    """95% Wilson score interval for a proportion -- better than the normal
    approximation near the edges, and what you should quote in the book."""
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total
                            + z * z / (4 * total * total))) / denom
    return (100.0 * (centre - margin), 100.0 * (centre + margin))


def per_class_accuracy(preds, labels):
    rows = []
    for c in range(10):
        mask = labels == c
        n = int(mask.sum())
        hit = int((preds[mask] == c).sum())
        rows.append({"class": CLASS_NAMES[c], "n": n,
                     "correct": hit,
                     "acc": round(100.0 * hit / n, 2) if n else 0.0})
    return rows


def mcnemar(preds_a, preds_b, labels):
    """McNemar test on paired classifier errors.

    b = A right, B wrong;  c = A wrong, B right.
    Under the null hypothesis (the two models are equally good) b and c come
    from the same binomial, so a large |b-c| relative to b+c is evidence of a
    real difference. Uses the continuity-corrected chi-square statistic.
    """
    a_ok = preds_a == labels
    b_ok = preds_b == labels
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())

    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0,
                "verdict": "identical predictions on every image"}

    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)

    # Two-sided p-value from the chi-square distribution with 1 dof, computed
    # via the error function so no scipy dependency is needed.
    p = math.erfc(math.sqrt(chi2 / 2.0))

    if p < 0.05:
        verdict = "difference IS statistically significant (p < 0.05)"
    else:
        verdict = "difference is NOT statistically significant (p >= 0.05)"

    return {"b_a_right_b_wrong": b, "c_a_wrong_b_right": c,
            "chi2": round(chi2, 3), "p_value": round(p, 4),
            "verdict": verdict}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compare FP32 and INT8/QAT VGG-Lite on the full CIFAR-10 test set.")
    ap.add_argument("--fp32-ckpt", default="vgg_lite_fp32_best.pth")
    ap.add_argument("--qat-ckpt", default="vgg_lite_int8_best.pth")
    ap.add_argument("--qat-module", default="vgg_lite_model",
                    help="Python module holding the Brevitas model (no .py)")
    ap.add_argument("--qat-class", default=None,
                    help="Class name inside that module")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--cifar-raw", default=None,
                    help="Path to the extracted cifar-10-batches-py folder "
                         "(use inside the FINN Docker container)")
    ap.add_argument("--save-predictions", default="predictions.npz",
                    help="Save per-image predictions and labels, so the board's "
                         "hardware predictions can be compared image-for-image")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="comparison_results.json")
    ap.add_argument("--skip-fp32", action="store_true",
                    help="Evaluate only the QAT model (use before FP32 training finishes)")
    args = ap.parse_args()

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 72)
    print("ACCURACY COMPARISON -- FULL CIFAR-10 TEST SET")
    print("=" * 72)
    print(f"device: {device}")
    print()

    loader = build_test_loader(args.data_root, args.cifar_raw)

    results = {}
    predictions = {}

    # --- QAT / INT8 model (the deployed one) -------------------------------
    print("loading QAT (INT8) model ...")
    qat_model, qat_meta = load_qat(args.qat_module, args.qat_class,
                                   args.qat_ckpt, device)
    print("evaluating QAT model on 10,000 images ...")
    p_qat, labels = collect_predictions(qat_model, loader, device)
    n_total = len(labels)
    qat_correct = int((p_qat == labels).sum())
    lo, hi = wilson_interval(qat_correct, n_total)
    results["qat_int8"] = {
        "meta": qat_meta,
        "correct": qat_correct,
        "total": n_total,
        "accuracy": round(100.0 * qat_correct / n_total, 2),
        "ci95": [round(lo, 2), round(hi, 2)],
        "per_class": per_class_accuracy(p_qat, labels),
    }
    predictions["qat_int8"] = p_qat
    print(f"  QAT  : {qat_correct}/{n_total} = "
          f"{100.0*qat_correct/n_total:.2f}%  (95% CI {lo:.2f}-{hi:.2f})")
    print()

    # --- FP32 reference ---------------------------------------------------
    if not args.skip_fp32:
        print("loading FP32 reference model ...")
        fp32_model, fp32_meta = load_fp32(args.fp32_ckpt, device)
        print("evaluating FP32 model on 10,000 images ...")
        p_fp32, _ = collect_predictions(fp32_model, loader, device)
        fp32_correct = int((p_fp32 == labels).sum())
        lo, hi = wilson_interval(fp32_correct, n_total)
        results["fp32_reference"] = {
            "meta": fp32_meta,
            "correct": fp32_correct,
            "total": n_total,
            "accuracy": round(100.0 * fp32_correct / n_total, 2),
            "ci95": [round(lo, 2), round(hi, 2)],
            "per_class": per_class_accuracy(p_fp32, labels),
        }
        predictions["fp32_reference"] = p_fp32
        print(f"  FP32 : {fp32_correct}/{n_total} = "
              f"{100.0*fp32_correct/n_total:.2f}%  (95% CI {lo:.2f}-{hi:.2f})")
        print()

    # --- The comparison ---------------------------------------------------
    print("=" * 72)
    print("SUMMARY TABLE FOR THE PROJECT BOOK")
    print("=" * 72)
    print(f"{'model':<26}{'correct':>10}{'accuracy':>11}{'95% CI':>18}")
    print("-" * 72)
    order = ["fp32_reference", "qat_int8"]
    labels_pretty = {
        "fp32_reference": "FP32 reference",
        "qat_int8": "INT8 (QAT, deployed)",
    }
    for key in order:
        if key not in results:
            continue
        r = results[key]
        ci = f"{r['ci95'][0]:.2f} - {r['ci95'][1]:.2f}"
        print(f"{labels_pretty[key]:<26}{r['correct']:>6}/{r['total']:<4}"
              f"{r['accuracy']:>10.2f}%{ci:>18}")
    print("-" * 72)

    if "fp32_reference" in results:
        delta = results["qat_int8"]["accuracy"] - results["fp32_reference"]["accuracy"]
        agree = float((predictions["fp32_reference"] == predictions["qat_int8"]).mean())
        mc = mcnemar(predictions["fp32_reference"], predictions["qat_int8"], labels)

        results["comparison"] = {
            "delta_int8_minus_fp32": round(delta, 2),
            "prediction_agreement_pct": round(100.0 * agree, 2),
            "mcnemar": mc,
        }

        print()
        print(f"quantization effect  : {delta:+.2f} percentage points "
              f"(INT8 minus FP32)")
        print(f"prediction agreement : {100.0*agree:.2f}% of the 10,000 images")
        print(f"McNemar              : chi2 = {mc['chi2']}, p = {mc['p_value']}")
        print(f"                       {mc['verdict']}")
        print()
        if mc["p_value"] >= 0.05:
            print("HOW TO WRITE THIS UP:")
            print("  The gap is within statistical noise. The defensible claim is")
            print("  'quantization to INT8 did not cost measurable accuracy', NOT")
            print("  'the quantized model is better' (or worse). State the McNemar")
            print("  p-value in the book -- it pre-empts the obvious question.")
        else:
            print("HOW TO WRITE THIS UP:")
            print("  The gap is statistically significant. Report the direction and")
            print("  the size honestly, with the p-value, and say which model wins.")

    print()
    print("PER-CLASS ACCURACY")
    print("-" * 72)
    active = [k for k in order if k in results]
    header = f"{'class':<14}" + "".join(f"{labels_pretty[k]:>28}" for k in active)
    print(header)
    for i in range(10):
        row = f"{CLASS_NAMES[i]:<14}"
        for key in active:
            pc = results[key]["per_class"][i]
            cell = f"{pc['correct']}/{pc['n']} ({pc['acc']:.1f}%)"
            row += f"{cell:>28}"
        print(row)
    print("=" * 72)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults written to {args.out}")

    # Per-image predictions, for the image-for-image check against the board.
    if args.save_predictions:
        np.savez_compressed(args.save_predictions,
                            labels=labels,
                            **{k: v for k, v in predictions.items()})
        print(f"per-image predictions written to {args.save_predictions}")
        print("  -> copy this to the board and compare it against the hardware")
        print("     predictions to verify functional equivalence on all 10,000.")
    print("\nREMINDER: the hardware accuracy number must come from the board, not")
    print("from here. Run the 10,000-image evaluation on the PYNQ and confirm the")
    print("hardware predictions match the QAT predictions above image-for-image.")


if __name__ == "__main__":
    main()
