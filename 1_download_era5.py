import cdsapi

dataset = "reanalysis-era5-land-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": [
        "2m_dewpoint_temperature",
        "2m_temperature",
        "surface_latent_heat_flux",
        "surface_net_solar_radiation",
        "surface_net_thermal_radiation",
        "surface_sensible_heat_flux",
        "runoff",
        "total_evaporation",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "surface_pressure",
        "total_precipitation"
    ],
    "year": ["2015","2016","2017","2018","2019","2020",
             "2021","2022","2023","2024","2025","2026"],
    "month": ["01","02","03","04","05","06",
              "07","08","09","10","11","12"],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [38, -9, 19, 12]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()