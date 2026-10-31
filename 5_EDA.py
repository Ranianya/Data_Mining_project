
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

INPUT_FILE = "era5_algeria_clean.csv"
OUTPUT_DIR = "results/eda"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

CLIMATE_VARIABLES = [
    "t2m", "d2m", "slhf", "ssr", "str", "sshf",
    "ro", "e", "u10", "v10", "sp", "tp"
]

VARIABLE_NAMES = {
    "t2m": "2m Temperature",
    "d2m": "2m Dewpoint Temperature",
    "slhf": "Surface Latent Heat Flux",
    "ssr": "Surface Net Solar Radiation",
    "str": "Surface Net Thermal Radiation",
    "sshf": "Surface Sensible Heat Flux",
    "ro": "Runoff",
    "e": "Total Evaporation",
    "u10": "10m U Wind Component",
    "v10": "10m V Wind Component",
    "sp": "Surface Pressure",
    "tp": "Total Precipitation"
}


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {filename}")


# ============================================================
# STEP 2: LOAD AND VERIFY DATASET
# ============================================================

print("\n========== LOADING DATASET ==========")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["valid_time"]
)

required_columns = CLIMATE_VARIABLES + [
    "valid_time", "latitude", "longitude",
    "year", "month", "cell_id"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

df = df.sort_values(
    ["cell_id", "valid_time"]
).reset_index(drop=True)

print("Dataset shape:", df.shape)
print("Start date:", df["valid_time"].min())
print("End date:", df["valid_time"].max())
print("Geographic cells:", df["cell_id"].nunique())
print("Unique months:", df["valid_time"].nunique())

print("\nFirst observations:")
print(df.head())

# ============================================================
# STEP 3: DATA QUALITY ANALYSIS
# ============================================================

print("\n========== DATA QUALITY ==========")

missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": (
        df.isna().mean() * 100
    ).round(3)
})

print(missing)

missing.to_csv(
    os.path.join(OUTPUT_DIR, "missing_values.csv")
)

duplicates = df.duplicated(
    subset=["cell_id", "valid_time"]
).sum()

print("\nDuplicate observations:", duplicates)

plt.figure(figsize=(12, 5))

missing[missing.index.isin(CLIMATE_VARIABLES)][
    "missing_percent"
].plot(kind="bar")

plt.title("Missing Values by Climate Variable")
plt.xlabel("Climate Variable")
plt.ylabel("Missing Values (%)")

save_plot("01_missing_values.png")

# ============================================================
# STEP 4: DESCRIPTIVE STATISTICS
# ============================================================

print("\n========== DESCRIPTIVE STATISTICS ==========")

statistics = df[CLIMATE_VARIABLES].describe().T

statistics["median"] = df[
    CLIMATE_VARIABLES
].median()

statistics["variance"] = df[
    CLIMATE_VARIABLES
].var()

statistics["skewness"] = df[
    CLIMATE_VARIABLES
].skew()

statistics["kurtosis"] = df[
    CLIMATE_VARIABLES
].kurt()

print(statistics.round(3))

statistics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "descriptive_statistics.csv"
    )
)

# ============================================================
# STEP 5: UNIT CONVERSIONS FOR EXPLORATION
# ============================================================

print("\n========== UNIT CONVERSIONS ==========")

# Precipitation: meters -> millimeters
df["tp_mm"] = df["tp"] * 1000

# Runoff: meters -> millimeters
df["ro_mm"] = df["ro"] * 1000

# Surface pressure: Pa -> hPa
df["sp_hpa"] = df["sp"] / 100

# Wind speed calculated from U and V components
df["wind_speed"] = np.sqrt(
    df["u10"] ** 2 + df["v10"] ** 2
)

print("Created: tp_mm, ro_mm, sp_hpa, wind_speed")

# ============================================================
# STEP 6: DISTRIBUTION OF CLIMATE VARIABLES
# ============================================================

print("\n========== VARIABLE DISTRIBUTIONS ==========")

plot_variables = [
    "t2m", "d2m", "slhf", "ssr", "str", "sshf",
    "ro_mm", "e", "u10", "v10", "sp_hpa", "tp_mm"
]

fig, axes = plt.subplots(
    3, 4,
    figsize=(20, 14)
)

axes = axes.flatten()

for i, variable in enumerate(plot_variables):
    sns.histplot(
        df[variable].dropna(),
        bins=40,
        ax=axes[i],
        color="steelblue"
    )

    axes[i].set_title(variable)
    axes[i].set_xlabel(variable)
    axes[i].set_ylabel("Frequency")

plt.suptitle(
    "Distribution of Climate Variables",
    fontsize=18
)

save_plot("02_variable_distributions.png")

# ============================================================
# STEP 7: TEMPERATURE DISTRIBUTION
# ============================================================

print("\n========== TEMPERATURE DISTRIBUTION ==========")

plt.figure(figsize=(10, 6))

sns.histplot(
    df["t2m"],
    bins=50,
    kde=True,
    color="tomato"
)

plt.axvline(
    df["t2m"].mean(),
    color="blue",
    linestyle="--",
    label=f"Mean: {df['t2m'].mean():.2f} °C"
)

plt.axvline(
    df["t2m"].median(),
    color="green",
    linestyle="--",
    label=f"Median: {df['t2m'].median():.2f} °C"
)

plt.title("Distribution of 2m Temperature in Algeria")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.legend()

save_plot("03_temperature_distribution.png")

# ============================================================
# STEP 8: OUTLIER DETECTION (IQR)
# ============================================================

print("\n========== OUTLIER DETECTION ==========")

outlier_results = []

for variable in CLIMATE_VARIABLES:
    series = df[variable].dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = (
        (series < lower) |
        (series > upper)
    ).sum()

    outlier_results.append({
        "variable": variable,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": outliers,
        "outlier_percent": (
            outliers / len(series) * 100
            if len(series) else np.nan
        )
    })

outlier_df = pd.DataFrame(outlier_results)

print(outlier_df.round(3))

outlier_df.to_csv(
    os.path.join(OUTPUT_DIR, "outlier_summary.csv"),
    index=False
)

plt.figure(figsize=(10, 5))

sns.boxplot(
    x=df["t2m"],
    color="salmon"
)

plt.title("Temperature Boxplot")
plt.xlabel("2m Temperature (°C)")

save_plot("04_temperature_boxplot.png")

# ============================================================
# STEP 9: CORRELATION ANALYSIS
# ============================================================

print("\n========== CORRELATION ANALYSIS ==========")

correlation = df[CLIMATE_VARIABLES].corr(
    method="pearson"
)

print(correlation.round(2))

correlation.to_csv(
    os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
)

plt.figure(figsize=(12, 10))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True
)

plt.title("Correlation Matrix of Climate Variables")

save_plot("05_correlation_heatmap.png")

# ============================================================
# STEP 10: MONTHLY TEMPERATURE ANALYSIS
# ============================================================

print("\n========== MONTHLY TEMPERATURE ==========")

monthly_temperature = (
    df.groupby("month")["t2m"]
    .agg(["mean", "std", "min", "max"])
)

print(monthly_temperature.round(2))

monthly_temperature.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "monthly_temperature.csv"
    )
)

plt.figure(figsize=(11, 6))

sns.lineplot(
    data=df.groupby(
        "month", as_index=False
    )["t2m"].mean(),
    x="month",
    y="t2m",
    marker="o"
)

plt.title("Average Temperature by Month")
plt.xlabel("Month")
plt.ylabel("Average Temperature (°C)")
plt.xticks(range(1, 13))

save_plot("06_monthly_temperature.png")

# ============================================================
# STEP 11: ANNUAL TEMPERATURE ANALYSIS
# ============================================================

print("\n========== ANNUAL TEMPERATURE ==========")

yearly_temperature = (
    df.groupby("year")["t2m"]
    .agg(["mean", "std", "min", "max", "count"])
)

print(yearly_temperature.round(2))

yearly_temperature.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "yearly_temperature.csv"
    )
)

plt.figure(figsize=(11, 6))

sns.lineplot(
    data=yearly_temperature.reset_index(),
    x="year",
    y="mean",
    marker="o"
)

plt.title("Average Annual Temperature in Algeria")
plt.xlabel("Year")
plt.ylabel("Average Temperature (°C)")

save_plot("07_annual_temperature.png")

# ============================================================
# STEP 12: TEMPERATURE TIME SERIES
# ============================================================

print("\n========== TEMPERATURE TIME SERIES ==========")

time_series = (
    df.groupby("valid_time")["t2m"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(14, 6))

sns.lineplot(
    data=time_series,
    x="valid_time",
    y="t2m"
)

plt.title("Monthly Temperature Evolution (2015–2026)")
plt.xlabel("Date")
plt.ylabel("Average Temperature (°C)")

save_plot("08_temperature_timeseries.png")

# ============================================================
# STEP 13: SPATIAL TEMPERATURE ANALYSIS
# ============================================================

print("\n========== SPATIAL TEMPERATURE ==========")

spatial_temperature = (
    df.groupby(
        ["latitude", "longitude"],
        as_index=False
    )["t2m"].mean()
)

plt.figure(figsize=(10, 9))

scatter = plt.scatter(
    spatial_temperature["longitude"],
    spatial_temperature["latitude"],
    c=spatial_temperature["t2m"],
    cmap="coolwarm",
    s=22,
    marker="s"
)

plt.colorbar(
    scatter,
    label="Average Temperature (°C)"
)

plt.title("Spatial Distribution of Temperature in Algeria")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.axis("equal")

save_plot("09_spatial_temperature.png")

# ============================================================
# STEP 14: YEAR-MONTH TEMPERATURE HEATMAP
# ============================================================

print("\n========== YEAR-MONTH HEATMAP ==========")

year_month = df.pivot_table(
    index="year",
    columns="month",
    values="t2m",
    aggfunc="mean"
)

plt.figure(figsize=(13, 7))

sns.heatmap(
    year_month,
    annot=True,
    fmt=".1f",
    cmap="coolwarm"
)

plt.title("Monthly Temperature by Year")
plt.xlabel("Month")
plt.ylabel("Year")

save_plot("10_year_month_heatmap.png")

# ============================================================
# STEP 15: FINAL SUMMARY
# ============================================================

print("\n========== EDA SUMMARY ==========")

print("Total observations:", len(df))
print("Geographic cells:", df["cell_id"].nunique())
print("Total months:", df["valid_time"].nunique())

print(
    "Mean temperature:",
    round(df["t2m"].mean(), 2),
    "°C"
)

print(
    "Median temperature:",
    round(df["t2m"].median(), 2),
    "°C"
)

print(
    "Temperature standard deviation:",
    round(df["t2m"].std(), 2),
    "°C"
)

print(
    "Temperature range:",
    round(
        df["t2m"].max() - df["t2m"].min(),
        2
    ),
    "°C"
)

print("\nEDA completed successfully!")
print("Results saved in:", OUTPUT_DIR)
