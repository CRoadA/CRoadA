import numpy as np
from rasterio.transform import from_origin
from rasterio import features
import geopandas as gpd

class Rasterizer():

    def get_rasterize_roads(self, gdf_edges, pixel_size = 1):
        min_x, min_y, max_x, max_y = gdf_edges.total_bounds
        width_m = int(max_x - min_x)
        height_m = int(max_y - min_y)

        print(height_m, width_m)
        transform = from_origin(int(min_x), int(max_y), pixel_size, pixel_size)

        grid = features.rasterize(
            [(geometry, 1) for geometry in gdf_edges.geometry],
            out_shape=(height_m, width_m), 
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )
        print(np.sum(grid)/(grid.shape[0] * grid.shape[1]))

        return grid
    
    
    # czy jeżeli podane współrzedne będą wychodziły poza obszar gdf_edges to czy ucinać ten fragment?
    def rasterize_fragment_from_coordinates(self, gdf_edges, min_x, max_x, min_y, max_y, pixel_size=1):
        width = int(max_x - min_x)
        height = int(max_y - min_y)
        transform = from_origin(int(min_x), int(max_y), pixel_size, pixel_size)

        grid = features.rasterize(
            [(geometry, 1) for geometry in gdf_edges.geometry],
            out_shape=(height, width), 
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )
        print(np.sum(grid)/(grid.shape[0] * grid.shape[1]))

        return grid
    
    # size - wymiary fragmentu - zakładam że jest to kwadrat 
    def rasterize_fragment_from_indexes(self, gdf_edges: gpd.GeoDataFrame, indexes : tuple, size : int, pixel_size=1) -> np.array:
        min_x, min_y, max_x, max_y = gdf_edges.total_bounds
        fragment_min_x = min_x + indexes[0] * size
        fragment_max_x = fragment_min_x + size
        fragment_min_y = min_y + indexes[1] * size
        fragment_max_y = fragment_min_y + size

        transform = from_origin(int(fragment_min_x), int(fragment_max_y), pixel_size, pixel_size)

        grid = features.rasterize(
            [(geometry, 1) for geometry in gdf_edges.geometry],
            out_shape=(size, size), 
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )
        print(np.sum(grid)/(grid.shape[0] * grid.shape[1]))

        return grid