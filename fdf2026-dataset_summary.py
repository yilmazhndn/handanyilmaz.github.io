import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

labels = pd.read_csv(os.path.join(BASE_DIR, "fdf_truth_labels.csv"))

n_events = len(labels)
n_gamma = (labels["particle"] == "gamma").sum()
n_neutron = (labels["particle"] == "neutron").sum()
n_pileup = labels["pileup_flag"].sum()
pileup_fraction = 100.0 * n_pileup / n_events

summary = pd.DataFrame([
    ["Total generated events", n_events],
    ["Gamma-like events", n_gamma],
    ["Neutron-like events", n_neutron],
    ["Samples per waveform", 512],
    ["Sampling interval", "4 ns"],
    ["Equivalent sampling rate", "250 MS/s"],
    ["ADC resolution", "12 bit"],
    ["Baseline", "1000 ADC counts"],
    ["HDL memory vectors", n_events],
    ["Pile-up events", int(n_pileup)],
    ["Pile-up fraction", f"{pileup_fraction:.1f}%"],
])

out_csv = os.path.join(BASE_DIR, "fdf_dataset_summary.csv")
out_txt = os.path.join(BASE_DIR, "fdf_dataset_summary.txt")

summary.to_csv(out_csv, index=False, header=["Item", "Value"])

with open(out_txt, "w") as f:
    f.write("FDF26 Geant4-driven waveform demonstrator dataset summary\n")
    f.write("---------------------------------------------------------\n")
    for _, row in summary.iterrows():
        f.write(f"{row[0]}: {row[1]}\n")

print(summary)
print("Saved:", out_csv)
print("Saved:", out_txt)
