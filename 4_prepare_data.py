
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import regionmask
import matplotlib.pyplot as plt

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

ds = xr.open_dataset("era5land_algeria.nc", engine="netcdf4")

# Remove unnecessary metadata variables
ds = ds.drop_vars(["expver", "number"], errors="ignore")

print("\n========== ORIGINAL DATASET ==========")
print(ds)
print("\nDimensions:", dict(ds.sizes))
print("\nVariables:", list(ds.data_vars))

# ============================================================
# STEP 2: CHECK VARIABLES AND CONVERT TEMPERATURE
# ============================================================

required_variables = [
    "t2m", "d2m", "slhf", "ssr", "str",
    "sshf", "ro", "e", "u10", "v10", "sp", "tp"
]

missing_variables = [
    var for var in required_variables
    if var not in ds.data_vars
]

if missing_variables:
    raise ValueError(
        f"Missing required variables: {missing_variables}"
    )

print("\n========== VARIABLE UNITS ==========")

for var in required_variables:
    print(
        f"{var}: "
        f"{ds[var].attrs.get('units', 'unknown')}"
    )

# Convert Kelvin to Celsius
for var in ["t2m", "d2m"]:
    ds[var] = ds[var] - 273.15
    ds[var].attrs["units"] = "degC"

print("\nTemperature converted to Celsius.")

# ============================================================
# STEP 3: CROP DATA TO ALGERIA
# ============================================================

gdf = gpd.read_file("dz.json")

print("\n========== GEOGRAPHIC DATA ==========")
print("Number of shapes:", len(gdf))
print("Original CRS:", gdf.crs)

if gdf.crs is None:
    raise ValueError(
        "The GeoJSON has no CRS. Verify its coordinate "
        "system before continuing."
    )

gdf = gdf.to_crs("EPSG:4326")

# Merge all geographic shapes
algeria = gdf.dissolve()

# Create geographic mask
mask = regionmask.mask_geopandas(
    algeria,
    ds["longitude"],
    ds["latitude"]
)

ds = ds.where(mask.notnull())

# Check valid cells
n_total = ds["t2m"].isel(valid_time=0).size

n_valid = int(
    ds["t2m"]
    .isel(valid_time=0)
    .notnull()
    .sum()
    .item()
)

print("\n========== ALGERIA MASK ==========")
print("Total grid cells:", n_total)
print("Cells inside Algeria:", n_valid)

if n_valid == 0:
    raise ValueError(
        "No valid cells inside Algeria. Check coordinates."
    )

# ============================================================
# STEP 4: REDUCE SPATIAL RESOLUTION BY FACTOR OF 4
# ============================================================

ds_small = ds.coarsen(
    latitude=4,
    longitude=4,
    boundary="pad"
).mean()

print("\n========== SPATIAL REDUCTION ==========")
print("Original dimensions:", dict(ds.sizes))
print("Reduced dimensions:", dict(ds_small.sizes))

n_valid_small = int(
    ds_small["t2m"]
    .isel(valid_time=0)
    .notnull()
    .sum()
    .item()
)

print(
    "Cells inside Algeria after reduction:",
    n_valid_small
)

# ============================================================
# STEP 5: VISUALIZE TEMPERATURE
# ============================================================

plt.figure(figsize=(10, 6))

ds_small["t2m"].isel(valid_time=0).plot(
    cmap="coolwarm"
)

plt.title("Algeria - Temperature at 2m (Reduced Resolution)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

# ============================================================
# STEP 6: CONVERT NETCDF TO DATAFRAME
# ============================================================

df = ds_small.to_dataframe().reset_index()

# Remove cells outside Algeria
df = df.dropna(subset=["t2m"]).copy()

# Extract temporal information
df["valid_time"] = pd.to_datetime(df["valid_time"])
df["year"] = df["valid_time"].dt.year
df["month"] = df["valid_time"].dt.month

# Create geographic cell identifier
df["cell_id"] = df.groupby(
    ["latitude", "longitude"]
).ngroup()

# Sort observations chronologically
df = df.sort_values(
    ["cell_id", "valid_time"]
).reset_index(drop=True)

print("\n========== DATAFRAME ==========")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ============================================================
# STEP 7: DATA QUALITY CHECKS
# ============================================================

print("\n========== DATE RANGE ==========")
print("Start date:", df["valid_time"].min())
print("End date:", df["valid_time"].max())

print("\n========== GEOGRAPHIC CELLS ==========")
print("Number of cells:", df["cell_id"].nunique())

print("\n========== MONTHS PER CELL ==========")

months_per_cell = df.groupby("cell_id").size()

print(months_per_cell.describe())
print(
    "Unique month counts:",
    months_per_cell.unique()
)

print("\n========== MISSING VALUES ==========")

missing_counts = df.isna().sum()
missing_percent = df.isna().mean() * 100

missing_summary = pd.DataFrame({
    "Missing Count": missing_counts,
    "Missing (%)": missing_percent.round(2)
})

print(missing_summary)

print("\n========== DUPLICATES ==========")

duplicates = df.duplicated(
    subset=["cell_id", "valid_time"]
).sum()

print("Duplicate observations:", duplicates)

print("\n========== DESCRIPTIVE STATISTICS ==========")

print(df[required_variables].describe().round(2))

print("\n========== TEMPERATURE CHECK ==========")

print("Minimum t2m:", df["t2m"].min())
print("Maximum t2m:", df["t2m"].max())

# ============================================================
# STEP 8: CHECK TEMPORAL CONTINUITY
# ============================================================

print("\n========== TEMPORAL CONTINUITY ==========")

df["month_index"] = (
    df["year"] * 12 + df["month"]
)

df["month_difference"] = (
    df.groupby("cell_id")["month_index"].diff()
)

gaps = df[
    df["month_difference"].notna()
    & (df["month_difference"] != 1)
]

print("Number of temporal gaps:", len(gaps))

if not gaps.empty:
    print(gaps[
        ["cell_id", "valid_time", "month_difference"]
    ].head(10))

# Remove temporary checking columns
df = df.drop(
    columns=["month_index", "month_difference"]
)

# ============================================================
# STEP 9: SAVE CLEAN DATA
# ============================================================

output_file = "era5_algeria_clean.csv"

df.to_csv(output_file, index=False)

print("\n========== EXPORT ==========")
print("Saved file:", output_file)
print("Final shape:", df.shape)

print("\nPreprocessing completed successfully!")
