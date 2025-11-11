import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv("/logbook.csv", sep=",")

# Extract Fitness_mean
fitness_values = df["Fitness_mean"].dropna().values

# Sturges rule
R = fitness_values.max() - fitness_values.min()
n = len(fitness_values)
k = int(1 + 3.322 * np.log10(n))
amplitude = R / k

#  bins
bins = np.arange(fitness_values.min(), fitness_values.max() + amplitude, amplitude)
freq, edges = np.histogram(fitness_values, bins=bins)

plt.figure(figsize=(7, 4))
for i in range(len(freq)):
    left = edges[i]
    width = edges[i+1] - edges[i]
    plt.bar(left, freq[i], width=width, align='edge', edgecolor="black")

plt.title("Histogram of Fitness_mean (manual bins)")
plt.xlabel("Fitness_mean")
plt.ylabel("Frequency")
plt.xticks(edges, rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

img_path = "/mnt/data/histogram_fitness_mean_manual.png"
plt.savefig(img_path, dpi=150)
plt.show()

print(f"[Download the histogram](sandbox:{img_path})")
