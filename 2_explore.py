import xarray as xr

ds = xr.open_dataset("era5land_algeria.nc")
print(ds)