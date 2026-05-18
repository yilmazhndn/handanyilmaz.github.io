#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poster-quality PSD plots for FDF/CERN poster.

Expected location:
    B4a_build/scripts/plot_clean_psd.py

Expected input file:
    B4a_build/fdf_truth_labels.csv

Outputs:
    B4a_build/figures/psd_scatter_clean_poster.png
    B4a_build/figures/psd_scatter_clean_poster.pdf
    B4a_build/figures/psd_hist_clean_poster.png
    B4a_build/figures/psd_hist_clean_poster.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LABELS_CSV = os.path.join(BASE_DIR, "fdf_truth_labels.csv")
FIG_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Poster style
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Scientific poster palette
COLOR_GAMMA = "#005F73"
COLOR_NEUTRON = "#AE2012"
COLOR_GRID = "#6C757D"
COLOR_TEXTBOX_EDGE = "#ADB5BD"

# -----------------------------------------------------------------------------
# Read data
# -----------------------------------------------------------------------------
if not os.path.exists(LABELS_CSV):
    raise FileNotFoundError(f"Missing labels file: {LABELS_CSV}")

labels = pd.read_csv(LABELS_CSV)

required_cols = [
    "event_id",
    "particle",
    "edep_total_MeV",
    "q_long",
    "psd",
    "pileup_flag",
]

missing_cols = [c for c in required_cols if c not in labels.columns]
if missing_cols:
    raise ValueError(f"Missing columns in labels CSV: {missing_cols}")

# -----------------------------------------------------------------------------
# Clean PSD selection
# -----------------------------------------------------------------------------
# Low-energy events are dominated by noise and are not useful for charge-comparison PSD.
# Pile-up events are excluded here to show the clean gamma/neutron PSD behavior.
clean = labels[
    (labels["edep_total_MeV"] > 0.10) &
    (labels["q_long"] > 200) &
    (labels["psd"] > -0.1) &
    (labels["psd"] < 0.9) &
    (labels["pileup_flag"] == 0)
].copy()

if clean.empty:
    raise RuntimeError("No clean PSD events found. Try relaxing the PSD selection cuts.")

gamma = clean[clean["particle"] == "gamma"].copy()
neutron = clean[clean["particle"] == "neutron"].copy()

if gamma.empty or neutron.empty:
    print("Warning: One particle class is empty after filtering.")
    print(clean["particle"].value_counts())

print("Total events:", len(labels))
print("Clean PSD events:", len(clean))
print("Particle counts:")
print(clean["particle"].value_counts())
print("\nPSD statistics:")
print(clean.groupby("particle")["psd"].describe())

gamma_mean = gamma["psd"].mean() if not gamma.empty else np.nan
neutron_mean = neutron["psd"].mean() if not neutron.empty else np.nan
delta_mean = neutron_mean - gamma_mean

# -----------------------------------------------------------------------------
# Helper function for consistent axes styling
# -----------------------------------------------------------------------------
def style_axes(ax):
    ax.grid(True, which="major", linestyle="-", linewidth=0.55, alpha=0.22)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.16)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


# -----------------------------------------------------------------------------
# 1) PSD scatter plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.8, 5.8))

if not gamma.empty:
    ax.scatter(
        gamma["edep_total_MeV"],
        gamma["psd"],
        s=28,
        alpha=0.68,
        color=COLOR_GAMMA,
        edgecolors="white",
        linewidths=0.35,
        label=f"Gamma-like events (n={len(gamma)})",
    )

if not neutron.empty:
    ax.scatter(
        neutron["edep_total_MeV"],
        neutron["psd"],
        s=28,
        alpha=0.68,
        color=COLOR_NEUTRON,
        edgecolors="white",
        linewidths=0.35,
        label=f"Neutron-like events (n={len(neutron)})",
    )

info_text = (
    f"Clean events: {len(clean)}\n"
    f"Pile-up excluded\n"
    f"Mean PSD separation: {delta_mean:.3f}"
)

ax.text(
    0.98,
    0.94,
    info_text,
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10.2,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor=COLOR_TEXTBOX_EDGE,
        alpha=0.94,
    ),
)

ax.set_title(
    "Charge-comparison PSD for filtered verification events",
    pad=12,
    weight="bold",
)

ax.set_xlabel("Deposited energy (MeV)")
ax.set_ylabel("PSD parameter")

style_axes(ax)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    frameon=True,
    fancybox=True,
    framealpha=0.96,
    borderpad=0.7,
    handlelength=2.2,
    columnspacing=1.8,
)

fig.subplots_adjust(bottom=0.28, top=0.88, left=0.10, right=0.98)

scatter_png = os.path.join(FIG_DIR, "psd_scatter_clean_poster.png")
scatter_pdf = os.path.join(FIG_DIR, "psd_scatter_clean_poster.pdf")

fig.savefig(scatter_png, bbox_inches="tight")
fig.savefig(scatter_pdf, bbox_inches="tight")
plt.close(fig)

# -----------------------------------------------------------------------------
# 2) PSD histogram plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.8, 5.8))

bins = np.linspace(
    max(-0.1, clean["psd"].min() - 0.02),
    min(0.9, clean["psd"].max() + 0.02),
    42,
)

if not gamma.empty:
    ax.hist(
        gamma["psd"],
        bins=bins,
        alpha=0.62,
        color=COLOR_GAMMA,
        edgecolor="white",
        linewidth=0.6,
        label=f"Gamma-like events (n={len(gamma)})",
    )

if not neutron.empty:
    ax.hist(
        neutron["psd"],
        bins=bins,
        alpha=0.62,
        color=COLOR_NEUTRON,
        edgecolor="white",
        linewidth=0.6,
        label=f"Neutron-like events (n={len(neutron)})",
    )

if not np.isnan(gamma_mean):
    ax.axvline(
        gamma_mean,
        color=COLOR_GAMMA,
        linestyle=(0, (5, 4)),
        linewidth=2.0,
        label=f"Gamma mean PSD = {gamma_mean:.3f}",
    )

if not np.isnan(neutron_mean):
    ax.axvline(
        neutron_mean,
        color=COLOR_NEUTRON,
        linestyle=(0, (5, 4)),
        linewidth=2.0,
        label=f"Neutron mean PSD = {neutron_mean:.3f}",
    )

info_text = (
    f"Energy cut: > 0.10 MeV\n"
    f"q_long cut: > 200\n"
    f"Pile-up flag: 0"
)

ax.text(
    0.98,
    0.94,
    info_text,
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10.2,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor=COLOR_TEXTBOX_EDGE,
        alpha=0.94,
    ),
)

ax.set_title(
    "PSD distribution for gamma- and neutron-like waveforms",
    pad=12,
    weight="bold",
)

ax.set_xlabel("PSD parameter")
ax.set_ylabel("Number of events")

style_axes(ax)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    frameon=True,
    fancybox=True,
    framealpha=0.96,
    borderpad=0.7,
    handlelength=2.2,
    columnspacing=1.8,
)

fig.subplots_adjust(bottom=0.32, top=0.88, left=0.10, right=0.98)

hist_png = os.path.join(FIG_DIR, "psd_hist_clean_poster.png")
hist_pdf = os.path.join(FIG_DIR, "psd_hist_clean_poster.pdf")

fig.savefig(hist_png, bbox_inches="tight")
fig.savefig(hist_pdf, bbox_inches="tight")
plt.close(fig)

print("\nSaved:")
print(scatter_png)
print(scatter_pdf)
print(hist_png)
print(hist_pdf)
