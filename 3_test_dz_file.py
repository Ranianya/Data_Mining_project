import geopandas as gpd
import matplotlib.pyplot as plt

gdf = gpd.read_file("dz.json")
print(len(gdf), "shapes")
print(gdf.crs)

gdf.dissolve().plot()
plt.show()