#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_figures.py — figures 3, 5 and 6 for the defence deck and the project book.

Labels are in English on purpose: every term on them (MVAU, LUT, BRAM, DSP,
FMPadding) is an English technical term anyway, and it keeps the figures free of
right-to-left rendering problems. A Hebrew caption goes under the figure, in the
slide or the book, not inside the image.

Palette: the two validated categorical slots.
    software (ARM)   orange  #eb6834
    accelerator (PL) blue    #2a78d6
Checked with the palette validator: adjacent CVD dE 24.7, normal-vision 33.6,
both well clear of the floors, contrast >= 3:1 on the light surface.

Every number below comes from results_10k.json, post_synth_resources.json and
estimate_layer_cycles.json.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# ---- tokens ---------------------------------------------------------------
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
INK_MUTED = "#8a8985"
GRID      = "#e3e2de"

SW  = "#eb6834"      # software on the ARM
HW  = "#2a78d6"      # the FPGA accelerator
HW_FILL = "#e8f0fb"
SW_FILL = "#fdeee7"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
})

OUT = "out/figures"
os.makedirs(OUT, exist_ok=True)

def titles(ax, title, subtitle):
    """Title and subtitle as axes-relative text.

    set_title + a second text block kept colliding: the title's pad and the
    text's y are measured against different boxes. Placing both against
    transAxes keeps the gap fixed whatever the figure size.
    """
    ax.text(0, 1.155, title, transform=ax.transAxes, fontsize=13.5,
            fontweight="bold", color=INK, va="bottom")
    ax.text(0, 1.045, subtitle, transform=ax.transAxes, fontsize=9.2,
            color=INK_2, va="bottom")


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=200, bbox_inches="tight",
                    pad_inches=0.25)
    plt.close(fig)
    print(f"  {name}.png / .svg")


# ===========================================================================
# FIGURE 3 — what is inside the accelerator
# ===========================================================================
def figure3():
    fig, ax = plt.subplots(figsize=(11.8, 6.4))
    ax.set_xlim(0, 104); ax.set_ylim(0, 62); ax.axis("off")

    BW, BH, GAP = 17.5, 9.0, 4.0
    X0 = 9.0                      # left edge of the first block
    RIGHT_CH, LEFT_CH = 94.5, 4.5  # channels the wrap arrows travel in
    ROWS_Y = (48.0, 29.0, 10.0)   # bottom edge of each row
    cy = lambda y: y + BH / 2

    def block(x, y, title, sub=None, strong=False):
        ax.add_patch(FancyBboxPatch(
            (x, y), BW, BH, boxstyle="round,pad=0.35,rounding_size=1.1",
            linewidth=2.2 if strong else 1.4,
            edgecolor=HW, facecolor="#cfe0f6" if strong else HW_FILL))
        ax.text(x + BW/2, y + BH/2 + (2.0 if sub else 0), title,
                ha="center", va="center", fontsize=9.5,
                fontweight="bold" if strong else "normal", color=INK)
        if sub:
            ax.text(x + BW/2, y + BH/2 - 3.0, sub, ha="center", va="center",
                    fontsize=8, color=INK_2)

    def line(x1, y1, x2, y2, head=False):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>" if head else "-",
            mutation_scale=13, linewidth=1.5, color=INK_MUTED,
            shrinkA=0, shrinkB=0))

    rows = [
        [("FMPadding_0", None, False), ("ConvInpGen_0", "sliding window", False),
         ("MVAU_hls_0", "conv2 · 2,304 w", True), ("MaxPool_0", None, False)],
        [("FMPadding_1", None, False), ("ConvInpGen_1", "sliding window", False),
         ("MVAU_hls_1", "conv3 · 4,608 w", False)],
        [("FMPadding_2", None, False), ("ConvInpGen_2", "sliding window", False),
         ("MVAU_hls_2", "conv4 · 9,216 w", False), ("MaxPool_1", None, False)],
    ]

    row_end = []
    for y, blocks in zip(ROWS_Y, rows):
        x = X0
        for i, (t, sub, strong) in enumerate(blocks):
            block(x, y, t, sub, strong)
            if i < len(blocks) - 1:
                line(x + BW + 0.5, cy(y), x + BW + GAP - 0.5, cy(y), head=True)
            x += BW + GAP
        row_end.append(x - GAP)

    # the stage labels sit to the LEFT of the wrap channel, so the horizontal
    # leg of a wrap arrow never crosses them
    for i, y in enumerate(ROWS_Y):
        ax.text(1.1, cy(y), f"stage {i+1}", rotation=90, ha="center", va="center",
                fontsize=8.5, color=INK_MUTED, fontweight="bold")

    # input
    line(LEFT_CH, cy(ROWS_Y[0]), X0 - 0.5, cy(ROWS_Y[0]), head=True)
    ax.text(LEFT_CH, ROWS_Y[0] + BH + 1.3, "input from DMA   (32, 32, 16) UINT8",
            fontsize=8.4, color=INK_MUTED, style="italic", ha="left", va="bottom")

    # wrap arrows, and the tensor shape on the horizontal leg of each
    for i, shp in enumerate(["(16, 16, 16)", "(16, 16, 32)"]):
        y_from, y_to = cy(ROWS_Y[i]), cy(ROWS_Y[i+1])
        y_mid = (y_from + y_to) / 2
        line(row_end[i] + 0.5, y_from, RIGHT_CH, y_from)
        line(RIGHT_CH, y_from, RIGHT_CH, y_mid)
        line(RIGHT_CH, y_mid, LEFT_CH, y_mid)
        line(LEFT_CH, y_mid, LEFT_CH, y_to)
        line(LEFT_CH, y_to, X0 - 0.5, y_to, head=True)
        ax.text(50, y_mid + 1.6, shp, fontsize=8.4, color=INK_MUTED,
                style="italic", ha="center")

    # output
    line(row_end[2] + 0.5, cy(ROWS_Y[2]), RIGHT_CH + 4.5, cy(ROWS_Y[2]), head=True)
    ax.text(103.0, ROWS_Y[2] - 1.2, "output to DMA\n(8, 8, 32) UINT8",
            fontsize=8.4, color=INK_MUTED, style="italic", ha="right", va="top")

    # the bottleneck note sits directly over the block it describes
    mvau0_cx = X0 + 2 * (BW + GAP) + BW / 2
    ax.text(mvau0_cx, ROWS_Y[0] + BH + 1.3, "bottleneck — 393,216 cycles",
            ha="center", va="bottom", fontsize=9, color=SW, fontweight="bold")

    ax.text(0.8, 1.2,
            "Between every pair of blocks FINN inserts buffers automatically: "
            "18 StreamingFIFO and 5 DataWidthConverter, together 19.4 % of the LUTs.",
            fontsize=8.3, color=INK_MUTED)

    titles(ax, "Inside the accelerator — three convolution stages, one pipeline",
           "conv2, conv3 and conv4 run on the FPGA; each has its own hardware and "
           "they work on different images at the same time")
    save(fig, "fig3_accelerator_pipeline")


# ===========================================================================
# FIGURE 5 — where the time goes
# ===========================================================================
def figure5():
    fig, ax = plt.subplots(figsize=(11.0, 3.9))

    HEAD, ACC, TAIL = 20.35, 11.05, 8.60
    SW_ONLY = 97.96
    SYSTEM = HEAD + ACC + TAIL

    y_sw, y_sys, h = 0.80, 0.0, 0.46

    ax.barh(y_sw, SW_ONLY, height=h, color=SW, edgecolor=SURFACE, linewidth=2)
    left = 0.0
    for val, col in ((HEAD, SW), (ACC, HW), (TAIL, SW)):
        ax.barh(y_sys, val, left=left, height=h, color=col,
                edgecolor=SURFACE, linewidth=2)
        left += val

    ax.text(SW_ONLY / 2, y_sw, "whole network on the ARM   97.96 ms",
            ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    ax.text(HEAD / 2, y_sys, f"{HEAD:.1f}", ha="center", va="center",
            fontsize=9.5, color="white", fontweight="bold")
    ax.text(HEAD + ACC / 2, y_sys, f"{ACC:.1f}", ha="center", va="center",
            fontsize=9.5, color="white", fontweight="bold")
    ax.text(HEAD + ACC + TAIL / 2, y_sys, f"{TAIL:.1f}", ha="center", va="center",
            fontsize=9.5, color="white", fontweight="bold")

    ax.annotate("conv1 + quantiser\non the ARM", xy=(HEAD/2, y_sys - h/2),
                xytext=(HEAD/2 - 2, y_sys - 0.46), ha="center", fontsize=8.6,
                color=INK_2, arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=1))
    ax.annotate("FPGA\naccelerator", xy=(HEAD + ACC/2, y_sys - h/2),
                xytext=(HEAD + ACC/2 + 0.5, y_sys - 0.76), ha="center", fontsize=8.6,
                color=HW, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=1))
    ax.annotate("dequantise + pool + FC\non the ARM", xy=(HEAD + ACC + TAIL/2, y_sys - h/2),
                xytext=(HEAD + ACC + TAIL/2 + 12, y_sys - 0.46), ha="center", fontsize=8.6,
                color=INK_2, arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=1))

    ax.text(SYSTEM + 1.6, y_sys, f"{SYSTEM:.2f} ms   ×2.45 faster",
            va="center", fontsize=10.5, color=INK, fontweight="bold")

    ax.set_yticks([y_sys, y_sw])
    ax.set_yticklabels(["with accelerator", "software only"], fontsize=10)
    ax.set_xlabel("latency for one image  (ms)", fontsize=9.5)
    ax.set_xlim(0, 112); ax.set_ylim(-1.05, 1.18)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)

    titles(ax, "Where the time goes — 72 % of the system latency is still software",
           "the accelerator replaces 75.37 ms of software with 11.05 ms (×6.82), "
           "but conv1, the quantiser and the classifier stay on the ARM")
    save(fig, "fig5_time_breakdown")


# ===========================================================================
# FIGURE 6 — resource utilisation
# ===========================================================================
def figure6():
    fig, ax = plt.subplots(figsize=(10.0, 4.6))

    rows = [("LUT",    49816, 53200),
            ("FF",     43492, 106400),
            ("BRAM18",    29, 280),
            ("DSP",       22, 220)]
    names = [r[0] for r in rows]
    pct   = [100 * r[1] / r[2] for r in rows]
    y = range(len(rows))[::-1]

    ax.barh(list(y), [100] * len(rows), height=0.52, color="#f0efeb",
            edgecolor=SURFACE, linewidth=2)
    ax.barh(list(y), pct, height=0.52, color=HW, edgecolor=SURFACE, linewidth=2)

    for yi, (name, used, avail), p in zip(y, rows, pct):
        # a bar that nearly fills the track has no room for an outside label
        if p > 60:
            ax.text(p - 2.0, yi, f"{p:.1f} %", va="center", ha="right",
                    fontsize=10.5, color="white", fontweight="bold")
        else:
            ax.text(p + 2.0, yi, f"{p:.1f} %", va="center", fontsize=10.5,
                    color=INK, fontweight="bold")
        ax.text(106, yi, f"{used:,} / {avail:,}", va="center", fontsize=9,
                color=INK_MUTED)

    ax.annotate("the binding constraint",
                xy=(80, list(y)[0] - 0.30), xytext=(52, list(y)[0] - 0.70),
                fontsize=9.5, color=SW, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=SW, lw=1.5))

    ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("utilisation of the Zynq-7020  (%)", fontsize=9.5)
    ax.set_xlim(0, 140); ax.set_ylim(-0.80, len(rows) - 0.45)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)

    titles(ax, "The logic ran out; the memory barely moved",
           "a convolutional network on this FPGA is limited by LUTs, not by BRAM — "
           "19.4 % of the LUTs go to FIFOs alone")
    save(fig, "fig6_resources")


if __name__ == "__main__":
    print("writing figures:")
    figure3(); figure5(); figure6()
    print("done ->", OUT)
