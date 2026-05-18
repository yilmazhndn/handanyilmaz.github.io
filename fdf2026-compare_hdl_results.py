import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

labels_path = os.path.join(BASE_DIR, "fdf_truth_labels.csv")
hdl_path = os.path.join(BASE_DIR, "hdl_results.csv")

labels = pd.read_csv(labels_path)
hdl = pd.read_csv(hdl_path)

df = labels.merge(hdl, on="event_id", how="inner")

df["reference_peak_index"] = df["trigger_index_reference"].astype(int)
df["hdl_peak_index"] = df["hdl_peak_index"].astype(int)

df["index_difference"] = df["hdl_peak_index"] - df["reference_peak_index"]
df["match"] = df["index_difference"] == 0

DT_NS = 4.0
df["hdl_detected_time_ns"] = df["hdl_peak_index"] * DT_NS
df["timing_error_ns"] = df["hdl_detected_time_ns"] - df["true_hit_time_ns"]

n_total = len(df)
n_match = int(df["match"].sum())
match_percent = 100.0 * n_match / n_total

summary_lines = []
summary_lines.append("HDL waveform replay and reference comparison summary")
summary_lines.append("----------------------------------------------------")
summary_lines.append(f"Compared events: {n_total}")
summary_lines.append(f"Exact peak-index matches: {n_match}")
summary_lines.append(f"Exact match ratio: {match_percent:.2f}%")
summary_lines.append(f"Mean index difference: {df['index_difference'].mean():.4f} samples")
summary_lines.append(f"Max absolute index difference: {df['index_difference'].abs().max()} samples")
summary_lines.append(f"Mean timing error: {df['timing_error_ns'].mean():.3f} ns")
summary_lines.append(f"Std timing error: {df['timing_error_ns'].std():.3f} ns")
summary_lines.append(f"Mean timing error, gamma: {df[df['particle']=='gamma']['timing_error_ns'].mean():.3f} ns")
summary_lines.append(f"Mean timing error, neutron: {df[df['particle']=='neutron']['timing_error_ns'].mean():.3f} ns")
summary_lines.append(f"Pile-up events compared: {int(df['pileup_flag'].sum())}")

out_csv = os.path.join(BASE_DIR, "hdl_comparison_results.csv")
out_txt = os.path.join(BASE_DIR, "hdl_comparison_summary.txt")

df.to_csv(out_csv, index=False)

with open(out_txt, "w") as f:
    f.write("\n".join(summary_lines))

print("\n".join(summary_lines))
print("Saved:", out_csv)
print("Saved:", out_txt)
