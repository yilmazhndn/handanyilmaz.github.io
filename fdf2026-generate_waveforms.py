#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FDF26 poster-quality waveform generator.

Input:
    B4a_build/fdf_gamma_events.csv
    B4a_build/fdf_neutron_events.csv

Output:
    B4a_build/fdf_waveforms.csv
    B4a_build/fdf_truth_labels.csv
    B4a_build/hdl_vectors/waveform_XXXXXX.mem

Poster-quality figures:
    B4a_build/figures/example_gamma_neutron_waveform_poster.png
    B4a_build/figures/example_gamma_neutron_waveform_poster.pdf
    B4a_build/figures/psd_scatter_poster.png
    B4a_build/figures/psd_scatter_poster.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
np.random.seed(42)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GAMMA_CSV = os.path.join(BASE_DIR, "fdf_gamma_events.csv")
NEUTRON_CSV = os.path.join(BASE_DIR, "fdf_neutron_events.csv")

OUT_WAVEFORMS = os.path.join(BASE_DIR, "fdf_waveforms.csv")
OUT_LABELS = os.path.join(BASE_DIR, "fdf_truth_labels.csv")
HDL_DIR = os.path.join(BASE_DIR, "hdl_vectors")
FIG_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(HDL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Digitizer settings
# -----------------------------------------------------------------------------
N_SAMPLES = 512
DT_NS = 4.0
TIME_NS = np.arange(N_SAMPLES) * DT_NS

ADC_BITS = 12
ADC_MAX = 2**ADC_BITS - 1
BASELINE = 1000.0
NOISE_SIGMA = 4.0

# -----------------------------------------------------------------------------
# Pulse settings
# -----------------------------------------------------------------------------
T0_MEAN_NS = 120.0
T0_SIGMA_NS = 8.0

# PSD windows
SHORT_WINDOW_NS = 60.0
LONG_WINDOW_NS = 240.0
SHORT_SAMPLES = int(SHORT_WINDOW_NS / DT_NS)
LONG_SAMPLES = int(LONG_WINDOW_NS / DT_NS)

# Scale from deposited energy to ADC amplitude
AMPLITUDE_PER_MEV = 900.0

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

COLOR_GAMMA = "#005F73"
COLOR_NEUTRON = "#AE2012"
COLOR_BASELINE = "#6C757D"
COLOR_TEXTBOX_EDGE = "#ADB5BD"


# -----------------------------------------------------------------------------
# Physics / electronics demonstrator model
# -----------------------------------------------------------------------------
def scintillation_pulse(t, t0, amplitude, tau_fast, tau_slow, slow_fraction):
    """
    Fast + slow scintillation pulse model.

    The model is intentionally simple and verification-oriented. It is used to
    generate realistic-looking digitized waveforms with particle-dependent pulse
    shape differences for HDL/HLS testbench replay.
    """
    x = t - t0
    pulse = np.zeros_like(t)

    mask = x >= 0
    fast = (1.0 - slow_fraction) * np.exp(-x[mask] / tau_fast)
    slow = slow_fraction * np.exp(-x[mask] / tau_slow)

    pulse[mask] = amplitude * (fast + slow)
    return pulse


def make_waveform(event_id, particle, edep_mev):
    """
    Create one digitized ADC waveform from one Geant4 event.
    """

    edep_mev = max(float(edep_mev), 0.0)

    amplitude = edep_mev * AMPLITUDE_PER_MEV
    amplitude *= np.random.normal(1.0, 0.04)

    t0 = np.random.normal(T0_MEAN_NS, T0_SIGMA_NS)

    if particle == "gamma":
        tau_fast = 8.0
        tau_slow = 45.0
        slow_fraction = 0.08
    elif particle == "neutron":
        tau_fast = 8.0
        tau_slow = 90.0
        slow_fraction = 0.25
    else:
        tau_fast = 8.0
        tau_slow = 60.0
        slow_fraction = 0.15

    pulse = scintillation_pulse(
        TIME_NS,
        t0,
        amplitude,
        tau_fast=tau_fast,
        tau_slow=tau_slow,
        slow_fraction=slow_fraction,
    )

    # Random pile-up in 8% of events
    pileup_flag = int(np.random.rand() < 0.08)

    if pileup_flag:
        t0_2 = t0 + np.random.uniform(40.0, 180.0)
        amp2 = amplitude * np.random.uniform(0.25, 0.8)

        pulse += scintillation_pulse(
            TIME_NS,
            t0_2,
            amp2,
            tau_fast=tau_fast,
            tau_slow=tau_slow,
            slow_fraction=slow_fraction,
        )
    else:
        t0_2 = np.nan

    # Baseline drift
    drift_slope = np.random.normal(0.0, 0.002)
    baseline_drift = drift_slope * (TIME_NS - TIME_NS[0])

    # Add baseline and electronic noise
    analog = BASELINE + baseline_drift + pulse
    analog += np.random.normal(0.0, NOISE_SIGMA, size=N_SAMPLES)

    # ADC quantization
    adc = np.clip(np.round(analog), 0, ADC_MAX).astype(int)

    return adc, t0, t0_2, pileup_flag


def compute_psd(adc):
    """
    Charge-comparison PSD demonstrator.

    q_short and q_long are calculated from the peak sample. This simple reference
    is sufficient for verification-oriented HDL replay comparisons.
    """
    signal = adc.astype(float) - BASELINE

    trigger_index = int(np.argmax(signal))

    q_short = np.sum(signal[trigger_index:trigger_index + SHORT_SAMPLES])
    q_long = np.sum(signal[trigger_index:trigger_index + LONG_SAMPLES])

    if q_long > 0:
        psd = (q_long - q_short) / q_long
    else:
        psd = 0.0

    return trigger_index, q_short, q_long, psd


def export_mem(event_id, adc):
    """
    Export ADC samples as hexadecimal values for HDL/SystemVerilog testbench.
    """
    filename = f"waveform_{event_id:06d}.mem"
    path = os.path.join(HDL_DIR, filename)

    with open(path, "w") as f:
        for value in adc:
            f.write(f"{int(value):03X}\n")

    return filename


def load_events():
    """
    Load gamma and neutron Geant4 event tables and merge them into a mixed-field
    verification dataset.
    """
    if not os.path.exists(GAMMA_CSV):
        raise FileNotFoundError(f"Missing gamma input file: {GAMMA_CSV}")

    if not os.path.exists(NEUTRON_CSV):
        raise FileNotFoundError(f"Missing neutron input file: {NEUTRON_CSV}")

    gamma = pd.read_csv(GAMMA_CSV)
    neutron = pd.read_csv(NEUTRON_CSV)

    required_cols = ["event_id", "edep_abs_MeV", "edep_gap_MeV"]

    for name, df in [("gamma", gamma), ("neutron", neutron)]:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {name} input file: {missing}")

    gamma["particle"] = "gamma"
    neutron["particle"] = "neutron"

    gamma["edep_total_MeV"] = gamma["edep_abs_MeV"] + gamma["edep_gap_MeV"]
    neutron["edep_total_MeV"] = neutron["edep_abs_MeV"] + neutron["edep_gap_MeV"]

    gamma["source_event_id"] = gamma["event_id"]
    neutron["source_event_id"] = neutron["event_id"]

    gamma["event_id"] = np.arange(len(gamma))
    neutron["event_id"] = np.arange(len(neutron)) + len(gamma)

    events = pd.concat([gamma, neutron], ignore_index=True)

    # Shuffle to mimic a mixed radiation field
    events = events.sample(frac=1.0, random_state=42).reset_index(drop=True)
    events["event_id"] = np.arange(len(events))

    return events


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def style_axes(ax):
    ax.grid(True, which="major", linestyle="-", linewidth=0.55, alpha=0.22)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.16)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def save_figure(fig, png_path, pdf_path):
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_representative_waveforms(example_gamma, example_neutron, labels):
    """
    Poster-quality representative gamma and neutron waveform figure.
    """
    if example_gamma is None or example_neutron is None:
        print("Representative waveform figure skipped: missing gamma or neutron example.")
        return

    gamma_event_id, gamma_adc = example_gamma
    neutron_event_id, neutron_adc = example_neutron

    gamma_label = labels[labels["event_id"] == gamma_event_id].iloc[0]
    neutron_label = labels[labels["event_id"] == neutron_event_id].iloc[0]

    gamma_t0 = float(gamma_label["true_hit_time_ns"])
    neutron_t0 = float(neutron_label["true_hit_time_ns"])

    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    ax.plot(
        TIME_NS,
        gamma_adc,
        color=COLOR_GAMMA,
        linewidth=2.5,
        label=f"Gamma-like waveform, event {gamma_event_id}",
    )

    ax.plot(
        TIME_NS,
        neutron_adc,
        color=COLOR_NEUTRON,
        linewidth=2.5,
        label=f"Neutron-like waveform, event {neutron_event_id}",
    )

    ax.axhline(
        BASELINE,
        color=COLOR_BASELINE,
        linestyle=(0, (5, 4)),
        linewidth=1.4,
        label=f"Baseline ({BASELINE:.0f} ADC)",
    )

    ax.axvline(
        gamma_t0,
        color=COLOR_GAMMA,
        linestyle=(0, (1, 2)),
        linewidth=2.0,
        alpha=0.85,
    )

    ax.axvline(
        neutron_t0,
        color=COLOR_NEUTRON,
        linestyle=(0, (1, 2)),
        linewidth=2.0,
        alpha=0.85,
    )

    info_text = (
        f"Sampling: {DT_NS:.1f} ns\n"
        f"ADC resolution: {ADC_BITS} bit\n"
        f"Noise sigma: {NOISE_SIGMA:.1f} ADC"
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
        "Representative digitized scintillator waveforms",
        pad=12,
        weight="bold",
    )

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("ADC counts")

    style_axes(ax)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        borderpad=0.7,
        handlelength=2.7,
        columnspacing=1.5,
    )

    fig.subplots_adjust(bottom=0.30, top=0.88, left=0.10, right=0.98)

    png_path = os.path.join(FIG_DIR, "example_gamma_neutron_waveform_poster.png")
    pdf_path = os.path.join(FIG_DIR, "example_gamma_neutron_waveform_poster.pdf")

    save_figure(fig, png_path, pdf_path)

    print("Saved:", png_path)
    print("Saved:", pdf_path)


def plot_psd_scatter(labels):
    """
    Poster-quality PSD scatter plot.
    """
    clean = labels[
        (labels["q_long"] > 200) &
        (labels["psd"] > -0.1) &
        (labels["psd"] < 0.9) &
        (labels["edep_total_MeV"] > 0.10)
    ].copy()

    if clean.empty:
        print("PSD scatter skipped: no clean events after filtering.")
        return

    gamma = clean[clean["particle"] == "gamma"]
    neutron = clean[clean["particle"] == "neutron"]

    gamma_mean = gamma["psd"].mean() if not gamma.empty else np.nan
    neutron_mean = neutron["psd"].mean() if not neutron.empty else np.nan
    delta_mean = neutron_mean - gamma_mean

    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    if not gamma.empty:
        ax.scatter(
            gamma["edep_total_MeV"],
            gamma["psd"],
            s=26,
            alpha=0.66,
            color=COLOR_GAMMA,
            edgecolors="white",
            linewidths=0.35,
            label=f"Gamma-like events (n={len(gamma)})",
        )

    if not neutron.empty:
        ax.scatter(
            neutron["edep_total_MeV"],
            neutron["psd"],
            s=26,
            alpha=0.66,
            color=COLOR_NEUTRON,
            edgecolors="white",
            linewidths=0.35,
            label=f"Neutron-like events (n={len(neutron)})",
        )

    info_text = (
        f"Events shown: {len(clean)}\n"
        f"Energy cut: > 0.10 MeV\n"
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
        "Charge-comparison PSD demonstrator",
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

    png_path = os.path.join(FIG_DIR, "psd_scatter_poster.png")
    pdf_path = os.path.join(FIG_DIR, "psd_scatter_poster.pdf")

    save_figure(fig, png_path, pdf_path)

    print("Saved:", png_path)
    print("Saved:", pdf_path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    events = load_events()

    waveform_rows = []
    label_rows = []

    example_gamma = None
    example_neutron = None

    for _, row in events.iterrows():
        event_id = int(row["event_id"])
        particle = str(row["particle"])
        edep_mev = float(row["edep_total_MeV"])

        adc, t0, t0_2, pileup_flag = make_waveform(event_id, particle, edep_mev)
        trigger_index, q_short, q_long, psd = compute_psd(adc)

        mem_file = export_mem(event_id, adc)

        for sample_index, adc_value in enumerate(adc):
            waveform_rows.append({
                "event_id": event_id,
                "particle": particle,
                "sample_index": sample_index,
                "time_ns": TIME_NS[sample_index],
                "adc": int(adc_value),
            })

        label_rows.append({
            "event_id": event_id,
            "particle": particle,
            "source_event_id": int(row["source_event_id"]),
            "edep_total_MeV": edep_mev,
            "true_hit_time_ns": t0,
            "pileup_flag": pileup_flag,
            "second_hit_time_ns": t0_2,
            "trigger_index_reference": trigger_index,
            "q_short": q_short,
            "q_long": q_long,
            "psd": psd,
            "mem_file": mem_file,
        })

        if particle == "gamma" and example_gamma is None and edep_mev > 0.1:
            example_gamma = (event_id, adc)

        if particle == "neutron" and example_neutron is None and edep_mev > 0.1:
            example_neutron = (event_id, adc)

    waveforms = pd.DataFrame(waveform_rows)
    labels = pd.DataFrame(label_rows)

    waveforms.to_csv(OUT_WAVEFORMS, index=False)
    labels.to_csv(OUT_LABELS, index=False)

    plot_representative_waveforms(example_gamma, example_neutron, labels)
    plot_psd_scatter(labels)

    print("\nDone.")
    print(f"Events processed: {len(events)}")
    print(f"Waveform CSV: {OUT_WAVEFORMS}")
    print(f"Truth labels CSV: {OUT_LABELS}")
    print(f"HDL vectors: {HDL_DIR}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
