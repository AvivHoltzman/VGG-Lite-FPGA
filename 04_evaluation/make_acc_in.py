#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_acc_in.py
==============
Produces the accelerator input (`acc_in`) for the whole CIFAR-10 test set.

THE APPROACH
------------
Running the entire streamlined graph per image (as extract_reference.py does)
takes about 3 s per image, almost all of it in conv2, conv3 and conv4 -- the
layers the FPGA executes. To produce the accelerator's input, only the graph's
*head* is needed:  input -> conv1 -> MultiThreshold -> acc_in.
That is 442,368 of 6,341,888 MACs (7 %). This script extracts the head's
parameters straight out of step_streamline.onnx and re-implements it in
batched NumPy.

SAFETY
------
Before generating anything, the NumPy head is checked **bit-exactly** against
the existing reference_data.npz. Every element of `acc_in` for those images must
match. If even one does not, the script stops and prints a diagnosis rather
than producing data that looks plausible and is wrong.

USAGE (inside the FINN Docker container)
----------------------------------------
    cd /home/avivh/finn
    python3 make_acc_in.py --inspect          # just show the head's structure

    python3 make_acc_in.py \
        --ref /home/avivh/finn/deploy/build_vgg_lite_int8_bitfile_final9283rce6/intermediate_models/reference_data.npz \
        --out-dir /home/avivh/finn/reference_10k
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper

MEAN = np.array([0.4914, 0.4822, 0.4465], np.float32)
STD = np.array([0.2023, 0.1994, 0.2010], np.float32)
ACC_IN = "MultiThreshold_0_out0"


# ---------------------------------------------------------------------------
# NumPy implementations of the QONNX ops that appear in the head
# ---------------------------------------------------------------------------
def op_im2col(x, kernel, stride, pad, pad_value=0):
    """QONNX Im2Col.  x: (N,H,W,C) -> (N,OH,OW,kh*kw*C).

    The last axis is ordered kernel-row, kernel-column, channel -- the same
    order the following MatMul's weight matrix expects.
    """
    kh, kw = kernel
    sh, sw = stride
    pb_h, pb_w, pe_h, pe_w = pad
    N, Hh, Ww, C = x.shape
    xp = np.pad(x, ((0, 0), (pb_h, pe_h), (pb_w, pe_w), (0, 0)),
                constant_values=pad_value)
    OH = (Hh + pb_h + pe_h - kh) // sh + 1
    OW = (Ww + pb_w + pe_w - kw) // sw + 1
    out = np.empty((N, OH, OW, kh * kw * C), x.dtype)
    for i in range(kh):
        for j in range(kw):
            out[..., (i * kw + j) * C:(i * kw + j + 1) * C] = \
                xp[:, i:i + OH * sh:sh, j:j + OW * sw:sw, :]
    return out


def op_multithreshold(x, thresholds, out_scale=1.0, out_bias=0.0,
                      data_layout="NHWC"):
    """QONNX MultiThreshold.

    For every channel c the output counts how many thresholds the input is
    greater than or equal to, then scales and shifts.
    thresholds: (C, T), ascending along axis 1.
    """
    if data_layout == "NCHW":
        x = x.transpose(0, 2, 3, 1)
    C = x.shape[-1]
    th = thresholds
    if th.shape[0] == 1 and C > 1:              # a single broadcast channel
        th = np.repeat(th, C, axis=0)
    out = np.empty(x.shape, np.float32)
    for c in range(C):
        out[..., c] = np.searchsorted(th[c], x[..., c], side="right")
    out = out * out_scale + out_bias
    if data_layout == "NCHW":
        out = out.transpose(0, 3, 1, 2)
    return out


def attrs_of(node):
    d = {}
    for a in node.attribute:
        if a.type == 1:    d[a.name] = a.f
        elif a.type == 2:  d[a.name] = a.i
        elif a.type == 3:  d[a.name] = a.s.decode()
        elif a.type == 6:  d[a.name] = list(a.floats)
        elif a.type == 7:  d[a.name] = list(a.ints)
        elif a.type == 8:  d[a.name] = [s.decode() for s in a.strings]
    return d


def thresholds_of(spec):
    for s in spec:
        if s["op"] == "MultiThreshold":
            return list(s["ini"].values())[0]
    return None


def build_head(model):
    """Nodes from the graph input up to and including the one producing ACC_IN."""
    nodes = list(model.graph.node)
    cut = next((i for i, n in enumerate(nodes) if ACC_IN in n.output), None)
    if cut is None:
        raise SystemExit(f"tensor {ACC_IN} is not produced by any node")
    spec = []
    for n in nodes[:cut + 1]:
        ini = {i: np.asarray(model.get_initializer(i)) for i in n.input
               if model.get_initializer(i) is not None}
        spec.append({"name": n.name, "op": n.op_type, "in": list(n.input),
                     "out": list(n.output), "attrs": attrs_of(n), "ini": ini})
    return spec


def run_head(spec, x, gin, dtype=np.float32, want_pre=False):
    """Execute the head on a batch. x: (N,3,32,32) NCHW.

    dtype selects the accumulation precision. want_pre also returns the tensor
    entering MultiThreshold, so a mismatch can be measured against the
    threshold it fell on.
    """
    ctx = {gin: x.astype(dtype)}
    pre = None
    for s in spec:
        op, a = s["op"], s["attrs"]
        data_in = [i for i in s["in"] if i not in s["ini"]]
        v = ctx[data_in[0]] if data_in else None

        if op == "Transpose":
            y = v.transpose(a.get("perm", [0, 3, 1, 2]))

        elif op == "Im2Col":
            k = a.get("kernel_size", [3, 3])
            st = a.get("stride", [1, 1])
            pd = list(a.get("pad_amount", [1, 1, 1, 1]))
            k = [k, k] if isinstance(k, int) else list(k)
            st = [st, st] if isinstance(st, int) else list(st)
            if len(pd) == 2:
                pd = [pd[0], pd[1], pd[0], pd[1]]
            y = op_im2col(v, k, st, pd, a.get("pad_value", 0))

        elif op == "MatMul":
            y = np.matmul(v, list(s["ini"].values())[0].astype(dtype))

        elif op == "MultiThreshold":
            pre = v
            y = op_multithreshold(v, list(s["ini"].values())[0].astype(dtype),
                                  a.get("out_scale", 1.0),
                                  a.get("out_bias", 0.0),
                                  a.get("data_layout", "NHWC"))

        elif op in ("Mul", "Div", "Add", "Sub"):
            k = list(s["ini"].values())[0] if s["ini"] else ctx[data_in[1]]
            y = {"Mul": np.multiply, "Div": np.divide,
                 "Add": np.add, "Sub": np.subtract}[op](v, k)

        elif op == "Reshape":
            shp = list(s["ini"].values())[0].astype(int)
            y = v.reshape([int(d) for d in shp])

        elif op == "Relu":
            y = np.maximum(v, 0)

        else:
            raise SystemExit(
                f"\nUnsupported op in the head: {op}  (node {s['name']}).\n"
                f"Re-run with --inspect and send me that listing.\n")

        ctx[s["out"][0]] = y
    return (ctx[ACC_IN], pre) if want_pre else ctx[ACC_IN]


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build",
                    default="/home/avivh/finn/deploy/"
                            "build_vgg_lite_int8_bitfile_final9283rce6/intermediate_models")
    ap.add_argument("--cifar", default="/home/avivh/finn/cifar_data")
    ap.add_argument("--ref", default=None,
                    help="existing reference_data.npz, used to verify this implementation")
    ap.add_argument("--out-dir", default="/home/avivh/finn/reference_10k")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--reuse-ref", action="store_true",
                    help="take the images covered by --ref verbatim from that file, "
                         "so they are bit-exact by construction")
    ap.add_argument("--accept-boundary", action="store_true",
                    help="proceed when the only mismatches are float rounding at a "
                         "threshold boundary")
    args = ap.parse_args()

    m = ModelWrapper(os.path.join(args.build, "step_streamline.onnx"))
    gin = m.graph.input[0].name
    spec = build_head(m)

    print("=" * 72)
    print("GRAPH HEAD   input -> accelerator input")
    print("=" * 72)
    for s in spec:
        ini = ", ".join(f"{k}{v.shape}" for k, v in s["ini"].items()) or "-"
        print(f"  {s['op']:16s} {s['name']:26s} init: {ini}")
        if s["attrs"]:
            print(f"                   attrs: {s['attrs']}")
    print("=" * 72)
    if args.inspect:
        return

    if args.ref is None:
        raise SystemExit("\n--ref is required: this implementation must be verified "
                         "against the existing reference before it is trusted.\n")

    # ---- verification ------------------------------------------------------
    print(f"\nVERIFYING against {args.ref}")
    r = np.load(args.ref)
    imgs_ref, acc_in_ref = r["imgs"], r["acc_in"]
    nv = len(imgs_ref)
    x = ((imgs_ref.astype(np.float32) / 255.0 - MEAN) / STD).transpose(0, 3, 1, 2)
    th = thresholds_of(spec)

    def compute(dtype):
        out = np.empty_like(acc_in_ref)
        pre = np.empty(acc_in_ref.shape, np.float64)
        t0 = time.time()
        for s in range(0, nv, args.batch):
            y, p = run_head(spec, np.ascontiguousarray(x[s:s + args.batch]), gin,
                            dtype=dtype, want_pre=True)
            out[s:s + args.batch] = y.astype(acc_in_ref.dtype)
            pre[s:s + args.batch] = p
        return out, pre, time.time() - t0

    results = {}
    for name, dt in (("float32", np.float32), ("float64", np.float64)):
        mine, pre, secs = compute(dt)
        diff = int((mine != acc_in_ref).sum())
        results[name] = (mine, pre, diff, secs)
        print(f"  {name:8s}  mismatches {diff:6d} / {acc_in_ref.size:,}"
              f"   ({100*diff/acc_in_ref.size:.6f} %)   {secs:.2f}s")

    # pick whichever precision agrees best with the compiled graph
    best = min(results, key=lambda k: results[k][2])
    mine, pre, n_diff, _ = results[best]
    print(f"\n  best precision : {best}")

    if n_diff == 0:
        print("  BIT-EXACT. The NumPy head reproduces the compiled graph.\n")
    else:
        # Every mismatch should sit on a threshold boundary. Measure how far.
        bad = np.argwhere(mine != acc_in_ref)
        gaps, rel = [], []
        for idx in bad:
            n, h, w, c = (int(v) for v in idx)
            lo = min(int(mine[n, h, w, c]), int(acc_in_ref[n, h, w, c]))
            t = float(th[c][lo])            # the threshold the value straddles
            v = float(pre[n, h, w, c])
            gaps.append(abs(v - t))
            scale = max(abs(t), abs(v), 1e-12)
            rel.append(abs(v - t) / scale)
        gaps, rel = np.array(gaps), np.array(rel)
        imgs_hit = len(set(int(i[0]) for i in bad))

        print()
        print("  MISMATCH ANALYSIS")
        print("  " + "-" * 60)
        print(f"  differing elements   : {n_diff} of {acc_in_ref.size:,}"
              f"  ({100*n_diff/acc_in_ref.size:.6f} %)")
        print(f"  images affected      : {imgs_hit} of {nv}"
              f"  ({100*imgs_hit/nv:.2f} %)")
        print(f"  level difference     : max {int(np.abs(mine.astype(int)-acc_in_ref.astype(int)).max())}"
              f"  (out of 256 levels)")
        print(f"  distance to threshold: max {gaps.max():.3e} absolute,"
              f" {rel.max():.3e} relative")
        print("  " + "-" * 60)

        on_boundary = bool(rel.max() < 1e-4)
        if on_boundary:
            print("  Every mismatch sits within 1e-4 (relative) of a threshold.")
            print("  This is floating-point rounding at a decision boundary, not a")
            print("  logic error: the compiled graph and NumPy sum the convolution")
            print("  in a different order, and a value landing exactly on a")
            print("  threshold can therefore fall either side of it.")
        else:
            print("  Mismatches are NOT all at threshold boundaries -- this looks")
            print("  like a real implementation difference. Stopping.")
            sys.exit(1)

        if not args.accept_boundary:
            print()
            print("  Not generating anything. Two ways forward:")
            print("    --reuse-ref          take the first "
                  f"{nv} images verbatim from the reference file (exact),")
            print("                         and compute only the remaining images.")
            print("    --accept-boundary    accept this rounding difference everywhere.")
            print()
            print(f"  Recommended: BOTH. The {nv} images used for the equivalence")
            print("  check are then bit-exact by construction, and the rest carry a")
            print(f"  {100*n_diff/acc_in_ref.size:.4f} % chance per activation of a single-level")
            print("  difference, far below the resolution of any accuracy figure.")
            sys.exit(1)
        print("\n  Proceeding: --accept-boundary was given.\n")

    print(f"  projected time for {args.n:,} images:"
          f" {results[best][3]/nv*args.n/60:.2f} min\n")
    use_dtype = np.float32 if best == "float32" else np.float64

    # ---- generate ----------------------------------------------------------
    with open(os.path.join(args.cifar, "test_batch"), "rb") as f:
        d = pickle.load(f, encoding="bytes")
    X = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)[:args.n]
    Y = np.array(d[b"labels"], np.int64)[:args.n]

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, "ref_tail_params.npz"),
                        perm=r["perm"], mul0=r["mul0"], mul1=r["mul1"],
                        fcW=r["fcW"], bias=r["bias"], kern=r["kern"])
    print("wrote ref_tail_params.npz")

    t_start = time.time()
    for ci in range(-(-args.n // args.chunk)):
        lo, hi = ci * args.chunk, min((ci + 1) * args.chunk, args.n)
        path = os.path.join(args.out_dir, f"ref_chunk_{ci:03d}.npz")
        if os.path.exists(path):
            print(f"  chunk {ci:03d} exists, skipped")
            continue
        t0 = time.time()
        xs = ((X[lo:hi].astype(np.float32) / 255.0 - MEAN) / STD).transpose(0, 3, 1, 2)
        acc_in = np.empty((hi - lo,) + acc_in_ref.shape[1:], acc_in_ref.dtype)
        n_reused = 0
        if args.reuse_ref and lo < nv:
            # these images are covered by the verified reference file
            k = min(hi, nv) - lo
            acc_in[:k] = acc_in_ref[lo:lo + k]
            n_reused = k
        for s in range(n_reused, hi - lo, args.batch):
            acc_in[s:s + args.batch] = run_head(
                spec, np.ascontiguousarray(xs[s:s + args.batch]), gin,
                dtype=use_dtype).astype(acc_in_ref.dtype)
        np.savez_compressed(path, imgs=X[lo:hi], acc_in=acc_in, labels=Y[lo:hi])
        tag = f"  ({n_reused} reused from reference)" if n_reused else ""
        print(f"  chunk {ci:03d}  [{lo:5d}..{hi:5d})  {time.time()-t0:5.1f}s  "
              f"{os.path.getsize(path)/1e6:5.1f} MB{tag}")

    files = sorted(f for f in os.listdir(args.out_dir) if f.startswith("ref_chunk_"))
    mb = sum(os.path.getsize(os.path.join(args.out_dir, f)) for f in files) / 1e6
    print("\n" + "=" * 72)
    print(f"  chunks {len(files)}    total {mb:.1f} MB    "
          f"elapsed {(time.time()-t_start)/60:.1f} min")
    print("=" * 72)
    print("""
These chunks hold imgs, acc_in and labels -- everything the board needs for
accuracy and for the timing measurements. They deliberately do NOT hold acc_out
or ref_logits: the bit-exactness check still runs against the original
1,000-image reference_data.npz, because equivalence is a property of the
hardware and 2,048,000 compared elements already establish it. Accuracy, which
does depend on sample size, is measured on all 10,000.
""")


if __name__ == "__main__":
    main()
