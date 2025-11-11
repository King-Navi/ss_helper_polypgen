#Para boxplot
import pandas as pd
import numpy as np

df = pd.read_csv("/mnt/data/logbook.csv", sep=",")

dice_values = df["DiceMetric_mean"].dropna().values

dice_min = np.min(dice_values)
q1 = np.percentile(dice_values, 25)
median = np.percentile(dice_values, 50)
q3 = np.percentile(dice_values, 75)
dice_max = np.max(dice_values)

summary = {
    "min": dice_min,
    "Q1 (25%)": q1,
    "Q2 / median (50%)": median,
    "Q3 (75%)": q3,
    "max": dice_max,
}

summary
