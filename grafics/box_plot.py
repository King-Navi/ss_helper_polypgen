import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load logbook
df = pd.read_csv("/logbook.csv", sep=",")

# DiceMetric_mean
dice = df["DiceMetric_mean"].dropna().values

# summary
dice_min = np.min(dice)
q1 = np.percentile(dice, 25)
median = np.percentile(dice, 50)
q3 = np.percentile(dice, 75)
dice_max = np.max(dice)

# IQR and fences
iqr = q3 - q1 #range
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

# Outliers
outliers = dice[(dice < lower_fence) | (dice > upper_fence)]
outliers_sorted = np.sort(outliers)

# Plot boxplot
plt.figure(figsize=(4, 6))
plt.boxplot(dice, vert=True, showmeans=True)
plt.ylabel("DiceMetric_mean")
plt.title("Boxplot of DiceMetric_mean")
plt.tight_layout()

img_path = "/mnt/data/boxplot_dice_metric_mean.png"
plt.savefig(img_path, dpi=150)
plt.show()

print("Outliers (if any):")
for v in outliers_sorted:
    print(v)

print(f"\n[Download boxplot](sandbox:{img_path})")
