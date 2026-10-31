
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "era5_algeria_clean.csv"
OUTPUT_DIR = "results/eda"

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

VARIABLES = [
    "t2m", "d2m", "slhf", "ssr", "str", "sshf",
    "ro", "e", "u10", "v10", "sp", "tp"
]

df = pd.read_csv(INPUT_FILE, parse_dates=["valid_time"])
df = df.sort_values(["cell_id", "valid_time"])

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    print("Saved:", filename)


# ============================================================
# 1. CORRECT ANNUAL TEMPERATURE COMPARISON
# ============================================================

print("\n========== ANNUAL TEMPERATURE ==========")

# Select years containing all 12 months
complete_years = (
    df.groupby("year")["month"]
    .nunique()
)

complete_years = complete_years[
    complete_years == 12
].index

df_complete = df[df["year"].isin(complete_years)].copy()

annual = (
    df_complete.groupby("year")["t2m"]
    .mean()
    .reset_index(name="mean_temperature")
)

annual.to_csv(
    f"{OUTPUT_DIR}/annual_temperature_corrected.csv",
    index=False
)

plt.figure(figsize=(11, 6))
sns.lineplot(
    data=annual,
    x="year",
    y="mean_temperature",
    marker="o"
)
plt.title("Annual Temperature - Complete Years Only")
plt.xlabel("Year")
plt.ylabel("Mean Temperature (°C)")
save_plot("11_annual_temperature_corrected.png")

print(annual)


# ============================================================
# 2. CHECK REDUNDANCY BETWEEN SLHF AND EVAPORATION
# ============================================================

print("\n========== SLHF / EVAPORATION ==========")

corr = df["slhf"].corr(df["e"])

print("Pearson correlation:", corr)

slope, intercept = np.polyfit(
    df["e"],
    df["slhf"],
    1
)

predicted = slope * df["e"] + intercept

residual = df["slhf"] - predicted

print("Slope:", slope)
print("Intercept:", intercept)
print("Residual std:", residual.std())

plt.figure(figsize=(9, 6))

# Sample for faster visualization
sample = df.sample(
    n=min(10000, len(df)),
    random_state=42
)

sns.scatterplot(
    data=sample,
    x="e",
    y="slhf",
    alpha=0.25,
    s=12
)

plt.title(f"SLHF vs Evaporation (r = {corr:.5f})")
plt.xlabel("Evaporation (m water equivalent)")
plt.ylabel("Surface Latent Heat Flux (J/m²)")
save_plot("12_slhf_evaporation_relationship.png")


# ============================================================
# 3. PRECIPITATION ANALYSIS
# ============================================================

print("\n========== PRECIPITATION ANALYSIS ==========")

df["tp_mm"] = df["tp"] * 1000

zero_count = (df["tp_mm"] == 0).sum()
zero_percent = zero_count / len(df) * 100

print("Exact zero values:", zero_count)
print("Exact zero percentage:", round(zero_percent, 2))
print(df["tp_mm"].describe())

# Note: near-zero positive values are not exact zeros.

plt.figure(figsize=(10, 6))

sns.histplot(
    df["tp_mm"],
    bins=60,
    color="royalblue"
)

plt.title("Precipitation Distribution")
plt.xlabel("Precipitation (mm)")
plt.ylabel("Frequency")
save_plot("13_precipitation_distribution.png")

# Log transform for visualization only
plt.figure(figsize=(10, 6))

sns.histplot(
    np.log1p(df["tp_mm"].clip(lower=0)),
    bins=60,
    color="teal"
)

plt.title("Log-Transformed Precipitation Distribution")
plt.xlabel("log(1 + precipitation in mm)")
plt.ylabel("Frequency")
save_plot("14_precipitation_log_distribution.png")

# Analyze the percentage of near-zero precipitation
thresholds = [0, 0.1, 1, 5, 10]

precip_summary = pd.DataFrame({
    "threshold_mm": thresholds,
    "percent_at_or_below": [
        (df["tp_mm"] <= t).mean() * 100
        for t in thresholds
    ]
})

precip_summary.to_csv(
    f"{OUTPUT_DIR}/precipitation_summary.csv",
    index=False
)

print(precip_summary)


# ============================================================
# 4. OUTLIER ANALYSIS FOR ALL VARIABLES
# ============================================================

print("\n========== OUTLIER ANALYSIS ==========")

outlier_results = []

for var in VARIABLES:
    series = df[var].dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = ((series < lower) | (series > upper)).sum()

    outlier_results.append({
        "variable": var,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": count,
        "outlier_percent": count / len(series) * 100
    })

outliers = pd.DataFrame(outlier_results)

print(outliers.round(3))

outliers.to_csv(
    f"{OUTPUT_DIR}/outliers_detailed.csv",
    index=False
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=outliers,
    x="variable",
    y="outlier_percent",
    color="coral"
)

plt.title("IQR Outlier Percentage by Variable")
plt.xlabel("Climate Variable")
plt.ylabel("Outliers (%)")
save_plot("15_outlier_percentages.png")


# ============================================================
# 5. SEASONAL TEMPERATURE ANALYSIS
# ============================================================

print("\n========== SEASONAL ANALYSIS ==========")

season_map = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn"
}

df["season"] = df["month"].map(season_map)

season_order = ["Winter", "Spring", "Summer", "Autumn"]

# Use complete years to avoid partial-season bias
seasonal = (
    df_complete.assign(
        season=df_complete["month"].map(season_map)
    )
    .groupby("season")["t2m"]
    .agg(["mean", "std", "min", "max"])
    .reindex(season_order)
)

print(seasonal.round(2))

seasonal.to_csv(
    f"{OUTPUT_DIR}/seasonal_temperature.csv"
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=seasonal.index,
    y=seasonal["mean"],
    color="steelblue"
)

plt.title("Average Temperature by Season (Complete Years)")
plt.xlabel("Season")
plt.ylabel("Mean Temperature (°C)")
save_plot("16_seasonal_temperature.png")


# ============================================================
# 6. TEMPERATURE ANOMALIES AND TREND
# ============================================================

print("\n========== TEMPERATURE ANOMALIES ==========")

# Reference period: 2015-2025
baseline = df[
    df["year"].between(2015, 2025)
]

# Monthly climatology for each geographic cell
climatology = (
    baseline.groupby(["cell_id", "month"])["t2m"]
    .mean()
    .rename("climatology")
)

df_anomaly = df.join(
    climatology,
    on=["cell_id", "month"]
)

df_anomaly["temperature_anomaly"] = (
    df_anomaly["t2m"] - df_anomaly["climatology"]
)

monthly_anomaly = (
    df_anomaly.groupby("valid_time")[
        "temperature_anomaly"
    ]
    .mean()
    .reset_index()
)

monthly_anomaly.to_csv(
    f"{OUTPUT_DIR}/monthly_temperature_anomalies.csv",
    index=False
)

plt.figure(figsize=(14, 6))

sns.lineplot(
    data=monthly_anomaly,
    x="valid_time",
    y="temperature_anomaly"
)

plt.axhline(0, color="black", linestyle="--")

plt.title("Monthly Temperature Anomalies (2015-2025 Baseline)")
plt.xlabel("Date")
plt.ylabel("Temperature Anomaly (°C)")
save_plot("17_temperature_anomalies.png")

# Annual anomalies: complete years only
annual_anomaly = (
    df_anomaly[
        df_anomaly["year"].isin(complete_years)
    ]
    .groupby("year")["temperature_anomaly"]
    .mean()
    .reset_index()
)

annual_anomaly.to_csv(
    f"{OUTPUT_DIR}/annual_temperature_anomalies.csv",
    index=False
)

plt.figure(figsize=(11, 6))

sns.barplot(
    data=annual_anomaly,
    x="year",
    y="temperature_anomaly",
    color="steelblue"
)

plt.axhline(0, color="black", linestyle="--")
plt.title("Annual Temperature Anomalies")
plt.ylabel("Temperature Anomaly (°C)")
save_plot("18_annual_temperature_anomalies.png")

# Simple linear trend for descriptive exploration
slope, intercept = np.polyfit(
    annual_anomaly["year"],
    annual_anomaly["temperature_anomaly"],
    1
)

print("Descriptive trend:", round(slope, 4), "°C/year")
print("Equivalent per decade:", round(slope * 10, 3), "°C/decade")

# This slope is descriptive, not a significance test.


# ============================================================
# 7. GENERATE AUTOMATIC EDA SUMMARY
# ============================================================

print("\n========== GENERATING EDA SUMMARY ==========")

report = f"""
EXPLORATORY DATA ANALYSIS - ALGERIA ERA5-LAND
================================================

DATASET
-------
Observations: {len(df)}
Geographic cells: {df['cell_id'].nunique()}
Time range: {df['valid_time'].min()} to {df['valid_time'].max()}
Missing values: {df.isna().sum().sum()}

ANNUAL ANALYSIS
---------------
Complete years: {list(complete_years)}
Mean annual temperature calculated without partial years.

SLHF / EVAPORATION
------------------
Pearson correlation: {corr:.6f}
Regression slope: {np.polyfit(df['e'], df['slhf'], 1)[0]:.4f}

PRECIPITATION
-------------
Exact zero precipitation: {zero_percent:.2f}%
Mean precipitation: {df['tp_mm'].mean():.4f} mm
Maximum precipitation: {df['tp_mm'].max():.4f} mm

SEASONAL TEMPERATURE
--------------------
{seasonal['mean'].round(2).to_string()}

TEMPERATURE ANOMALIES
---------------------
Reference period: 2015-2025
Descriptive linear trend: {slope:.4f} degC/year
Equivalent: {slope * 10:.3f} degC/decade

INTERPRETATION NOTES
--------------------
1. Temperature displays strong seasonality.
2. Precipitation is highly skewed toward low values.
3. SLHF and evaporation show strong redundancy.
4. IQR outliers should not automatically be deleted.
5. The trend is descriptive and requires statistical
   validation before drawing climate conclusions.
6. 2026 is incomplete and excluded from annual comparisons.

NEXT STEP
---------
Feature Engineering for supervised and unsupervised learning.
"""

report_path = os.path.join(
    OUTPUT_DIR,
    "eda_summary.txt"
)

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print("\nSaved:", report_path)
print("\nEDA complementary analysis completed!")
