import numpy as np
from rasterio.transform import from_origin
from rasterio import features
import geopandas as gpd
import shapely

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

        return grid
    
    
    def rasterize_segment_from_coordinates(self, gdf_edges, min_x, max_x, min_y, max_y, pixel_size=1):
        gdfmin_x, gdfmin_y, gdfmax_x, gdfmax_y = gdf_edges.total_bounds
        real_min_x = max(min_x, gdfmin_x)
        real_max_x = min(max_x, gdfmax_x)
        real_max_y = min(max_y, gdfmax_y)
        real_min_y = max(min_y, gdfmin_y)
        width = int(real_max_x - real_min_x)
        height = int(real_max_y - real_min_y)
        transform = from_origin(int(real_min_x), int(real_max_y), pixel_size, pixel_size)

        segment_bounds = shapely.geometry.box(real_min_x, real_min_y, real_max_x, real_max_y)
        gdf_edges = gdf_edges.clip(segment_bounds)

        grid = features.rasterize(
            [(geometry, 1) for geometry in gdf_edges.geometry],
            out_shape=(height, width), 
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )

        return grid
    

    def rasterize_segment_from_indexes(self, gdf_edges: gpd.GeoDataFrame, indexes : tuple, size_w : int, size_h : int, pixel_size=1) -> np.array:
        min_x, min_y, max_x, max_y = gdf_edges.total_bounds
        segment_min_x = min_x + indexes[0] * size_w * pixel_size
        segment_max_x = segment_min_x + size_w * pixel_size
        if segment_max_x > max_x:
            segment_max_x = max_x
            size_w = int((segment_max_x - segment_min_x) / pixel_size)

        segment_min_y = min_y + indexes[1] * size_h * pixel_size
        segment_max_y = segment_min_y + size_h * pixel_size
        if segment_max_y > max_y:
            segment_max_y = max_y
            size_h = int((segment_max_y - segment_min_y) / pixel_size)

        transform = from_origin(int(segment_min_x), int(segment_max_y), pixel_size, pixel_size)

        segment_bounds = shapely.geometry.box(segment_min_x, segment_min_y, segment_max_x, segment_max_y)
        gdf_edges = gdf_edges.clip(segment_bounds)

        grid = features.rasterize(
            [(geometry, 1) for geometry in gdf_edges.geometry],
            out_shape=(size_w, size_h), # poprawić długości
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )

        return grid