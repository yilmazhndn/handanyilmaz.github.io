#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poster-quality pile-up corner-case plot for FDF/CERN poster.

Expected location:
    B4a_build/scripts/plot_pileup_case.py

Expected input files:
    B4a_build/fdf_truth_labels.csv
    B4a_build/fdf_waveforms.csv

Outputs:
    B4a_build/figures/pileup_corner_case_poster.png
    B4a_build/figures/pileup_corner_case_poster.pdf
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
WAVEFORMS_CSV = os.path.join(BASE_DIR, "fdf_waveforms.csv")
FIG_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# General poster style
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

# Poster-friendly scientific palette
COLOR_WAVEFORM = "#005F73"
COLOR_FILL = "#0A9396"
COLOR_BASELINE = "#6C757D"
COLOR_HIT1 = "#AE2012"
COLOR_HIT2 = "#EE9B00"
COLOR_MARKER = "#001219"

# -----------------------------------------------------------------------------
# Read data
# -----------------------------------------------------------------------------
if not os.path.exists(LABELS_CSV):
    raise FileNotFoundError(f"Missing labels file: {LABELS_CSV}")

if not os.path.exists(WAVEFORMS_CSV):
    raise FileNotFoundError(f"Missing waveforms file: {WAVEFORMS_CSV}")

labels = pd.read_csv(LABELS_CSV)
waveforms = pd.read_csv(WAVEFORMS_CSV)

required_label_cols = [
    "event_id",
    "pileup_flag",
    "edep_total_MeV",
    "particle",
    "true_hit_time_ns",
    "second_hit_time_ns",
]

required_waveform_cols = [
    "event_id",
    "time_ns",
    "adc",
]

missing_label_cols = [c for c in required_label_cols if c not in labels.columns]
missing_waveform_cols = [c for c in required_waveform_cols if c not in waveforms.columns]

if missing_label_cols:
    raise ValueError(f"Missing columns in labels CSV: {missing_label_cols}")

if missing_waveform_cols:
    raise ValueError(f"Missing columns in waveform CSV: {missing_waveform_cols}")

# -----------------------------------------------------------------------------
# Select a clear pile-up case
# -----------------------------------------------------------------------------
pileup_events = labels[
    (labels["pileup_flag"] == 1) &
    (labels["edep_total_MeV"] > 0.05)
].copy()

if pileup_events.empty:
    raise RuntimeError("No suitable pile-up event found. Try lowering the energy threshold.")

# Highest-energy pile-up case gives the clearest poster figure
selected = pileup_events.sort_values("edep_total_MeV", ascending=False).iloc[0]

EVENT_ID = int(selected["event_id"])
particle = str(selected["particle"])
t1 = float(selected["true_hit_time_ns"])
t2 = float(selected["second_hit_time_ns"])
edep = float(selected["edep_total_MeV"])

wf = waveforms[waveforms["event_id"] == EVENT_ID].copy().sort_values("time_ns")

if wf.empty:
    raise RuntimeError(f"No waveform samples found for event_id={EVENT_ID}")

# Robust baseline estimate from the first 10% of samples
n_baseline = max(5, int(0.10 * len(wf)))
baseline = float(np.median(wf["adc"].iloc[:n_baseline]))

peak_idx = int(wf["adc"].idxmax())
peak_time = float(wf.loc[peak_idx, "time_ns"])
peak_adc = float(wf.loc[peak_idx, "adc"])

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.8, 5.8))

ax.plot(
    wf["time_ns"],
    wf["adc"],
    color=COLOR_WAVEFORM,
    linewidth=2.6,
    solid_capstyle="round",
    label="Digitized detector waveform",
)

ax.fill_between(
    wf["time_ns"],
    baseline,
    wf["adc"],
    where=wf["adc"] >= baseline,
    color=COLOR_FILL,
    alpha=0.18,
    linewidth=0,
)

ax.axhline(
    baseline,
    color=COLOR_BASELINE,
    linestyle=(0, (5, 4)),
    linewidth=1.4,
    label=f"Estimated baseline ({baseline:.0f} ADC)",
)

ax.axvline(
    t1,
    color=COLOR_HIT1,
    linestyle=(0, (1, 2)),
    linewidth=2.4,
    label="Truth hit time 1",
)

ax.axvline(
    t2,
    color=COLOR_HIT2,
    linestyle=(0, (1, 2)),
    linewidth=2.4,
    label="Truth hit time 2",
)

ax.scatter(
    [peak_time],
    [peak_adc],
    s=48,
    color=COLOR_MARKER,
    zorder=5,
    label="Detected waveform peak",
)

# Information box: keep it inside the plot but away from the main pulse
info_text = (
    f"Event ID: {EVENT_ID}\n"
    f"Particle label: {particle}\n"
    f"Total deposited energy: {edep:.3f} MeV"
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
        edgecolor="#ADB5BD",
        alpha=0.94,
    ),
)

ax.set_title(
    "Pile-up corner case for FPGA waveform-replay verification",
    pad=12,
    weight="bold",
)

ax.set_xlabel("Time (ns)")
ax.set_ylabel("ADC counts")

ax.grid(True, which="major", linestyle="-", linewidth=0.55, alpha=0.22)
ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.16)

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# -----------------------------------------------------------------------------
# Important change:
# Legend is placed outside the plotting area, below the figure.
# This prevents the legend from covering the waveform and truth markers.
# -----------------------------------------------------------------------------
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=3,
    frameon=True,
    fancybox=True,
    framealpha=0.96,
    borderpad=0.7,
    handlelength=2.8,
    columnspacing=1.6,
)

# Leave space at the bottom for the external legend
fig.subplots_adjust(bottom=0.28, top=0.88, left=0.10, right=0.98)

out_png = os.path.join(FIG_DIR, "pileup_corner_case_poster.png")
out_pdf = os.path.join(FIG_DIR, "pileup_corner_case_poster.pdf")

fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

print("Selected pile-up event:")
print(selected.to_string())
print(f"Peak time: {peak_time:.2f} ns")
print(f"Peak ADC: {peak_adc:.1f}")
print("Saved:", out_png)
print("Saved:", out_pdf)
