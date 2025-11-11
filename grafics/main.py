# This script:
# 1) Reads the 5 fold metric files uploaded by the user
# 2) Extracts DiceMetric_mean and DiceMetric_std for each fold
# 3) Builds a histogram of DiceMetric_mean across folds
# 4) Simulates per-fold distributions (truncated to [0,1]) from mean/std and makes a boxplot (one box per fold)
# 5) Saves the figures and shows them inline; also displays a small summary table for reference

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

folder = Path("HERE")
files = [folder / f"data_fold{i}.txt" for i in range(1, 6)]

pattern_mean = re.compile(r"DiceMetric_mean\s*\n([0-9eE\.\-]+)")
pattern_std  = re.compile(r"DiceMetric_std\s*\n([0-9eE\.\-]+)")

fold_rows = []
for i, fp in enumerate(files, start=1):
    if not fp.exists():
        continue
    text = fp.read_text(encoding="utf-8", errors="ignore")
    m_mean = pattern_mean.search(text)
    m_std = pattern_std.search(text)
    dice_mean = float(m_mean.group(1)) if m_mean else np.nan
    dice_std  = float(m_std.group(1)) if m_std else np.nan
    fold_rows.append({"fold": i, "DiceMetric_mean": dice_mean, "DiceMetric_std": dice_std})

df = pd.DataFrame(fold_rows).set_index("fold").sort_index()

from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("Best individual - per-fold Dice summary", df.reset_index())

plt.figure(figsize=(6, 4))
means = df["DiceMetric_mean"].dropna().values
plt.hist(means, bins="auto", edgecolor="black")
plt.title("Histogram of DiceMetric_mean across folds")
plt.xlabel("DiceMetric_mean")
plt.ylabel("Count")
hist_path = str(folder / "histogram_dice_mean_across_folds.png")
plt.tight_layout()
plt.savefig(hist_path, dpi=150)
plt.show()

# --------- Figure 2: Boxplot ---------

sim_samples_per_fold = 1000
sim_data = []
labels = []

rng = np.random.default_rng(42)
for fold, row in df.iterrows():
    mu = row["DiceMetric_mean"]
    sd = row["DiceMetric_std"]
    if np.isnan(mu) or np.isnan(sd) or sd <= 0:
        samples = np.full(sim_samples_per_fold, mu if not np.isnan(mu) else 0.0)
    else:
        samples = rng.normal(mu, sd, size=sim_samples_per_fold)
        samples = np.clip(samples, 0.0, 1.0)
    sim_data.append(samples)
    labels.append(f"Fold {fold}")

plt.figure(figsize=(7, 4))
plt.boxplot(sim_data, labels=labels, showmeans=True)
plt.title("Per-fold Dice distribution (simulated from mean±std)")
plt.ylabel("Dice")
box_path = str(folder / "boxplot_dice_per_fold_simulated.png")
plt.tight_layout()
plt.savefig(box_path, dpi=150)
plt.show()

print(f"Saved files:\n- {hist_path}\n- {box_path}")
