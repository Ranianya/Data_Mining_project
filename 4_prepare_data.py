import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import regionmask

ds = xr.open_dataset("era5land_algeria.nc")
#drop unnecessary variables (metadata)
ds = ds.drop_vars(["expver", "number"], errors="ignore")

ds["t2m"] = ds["t2m"] - 273.15  # Convert temperature from Kelvin to Celsius
ds["d2m"] = ds["d2m"] - 273.15

# 3. Check
# print(ds)
# print("t2m min/max (°C):", float(ds["t2m"].min()), float(ds["t2m"].max()))
# print("d2m min/max (°C):", float(ds["d2m"].min()), float(ds["d2m"].max()))
# print("sp min/max (Pa):", float(ds["sp"].min()), float(ds["sp"].max())) 


# Step 3: crop to Algeria
gdf = gpd.read_file("dz.json")
algeria = gdf.dissolve()          # merge the 58 wilayas into one shape

mask = regionmask.mask_geopandas(algeria, ds["longitude"], ds["latitude"])
ds = ds.where(mask.notnull())     # outside Algeria -> NaN

# Check
n_total = ds["t2m"].isel(valid_time=0).size
n_valid = int(ds["t2m"].isel(valid_time=0).notnull().sum())
# print("cells in the box:", n_total)
# print("cells inside Algeria:", n_valid)

import matplotlib.pyplot as plt

ds["t2m"].isel(valid_time=6).plot()
# plt.show()

# Step 4: reduce spatial resolution by a factor of 4
ds_small = ds.coarsen(latitude=4, longitude=4, boundary="pad").mean()

# Check
print(ds_small.sizes)
n_valid_small = int(ds_small["t2m"].isel(valid_time=0).notnull().sum())
print("cells inside Algeria after reduction:", n_valid_small)

# Step 5: convert to a table (one row = one cell and one month)
df = ds_small.to_dataframe().reset_index() 
df = df.dropna(subset=["t2m"])        # remove cells outside Algeria

df["year"] = df["valid_time"].dt.year
df["month"] = df["valid_time"].dt.month
df["cell_id"] = df.groupby(["latitude", "longitude"]).ngroup()

df = df.sort_values(["cell_id", "valid_time"]).reset_index(drop=True)

# Step 6: quality checks
print(df.shape)
print(df["valid_time"].min(), df["valid_time"].max())
print("cells:", df["cell_id"].nunique())
print("months per cell:", df.groupby("cell_id").size().unique())
print(df.isna().sum())

# save to new file
df.to_csv("era5_algeria_clean.csv", index=False)
print("saved:", df.shape)