from grid_manager import GridManager
from scraper.grid_builder import GridBuilder 
from scraper.rasterizer import Rasterizer
import numpy as np
import math
import os
import srtm

class DataLoader():

    grid_density: float
    segment_h: int
    segment_w: int
    data_dir: str

    def __init__(self, grid_density: float, segment_h: int = 5000, segment_w: int = 5000, data_dir: str = "grids"):
        """Create DataLoader object.
        Args:
            grid_density (float): Distance between two closest different points on the grid (in meters).
            segment_h (int): Height of segment (in grid rows).
            segment_w (int): Width of segment (in grid columns).
            data_dir (str): Folder for the target files."""
        self.grid_density = grid_density
        self.segment_h = segment_h
        self.segment_w = segment_w
        self.data_dir = data_dir


    def load_city_grid(self, city: str, file_name: str) -> GridManager:
        """Load city grid to a given file.
        Args:
            city (str): String for identification of the city (OSM-like).
            file_name (str): Target file name.
        Returns:
            grid_manager (GridManager): Object handling partial load/write to the specified file.
        Raises:
            FileExistsError: if file with specified name already exists.
            """
        if os.path.exists(os.path.join(self.data_dir, file_name)):
            raise FileExistsError(f"File: {file_name} already exists in {self.data_dir} directory")
        
        builder = GridBuilder()
        gdf_edges = builder.get_city_roads(city)
        min_x, min_y, max_x, max_y = gdf_edges.total_bounds
        rows_number = max_x - min_x
        columns_number = max_y - min_y

        segment_rows = math.ceil((rows_number)/(self.segment_w * self.grid_density))
        segment_cols = math.ceil((columns_number)/(self.segment_h * self.grid_density))
        grid_manager = GridManager(file_name, rows_number=int(rows_number), columns_number=int(columns_number), 
                                   grid_density = self.grid_density, segment_h=self.segment_h, segment_w=self.segment_w, 
                                   data_dir=self.data_dir, upper_left_longitude=max_x, upper_left_latitude=max_y)
        print(f"Width: {int(rows_number)}, height: {int(columns_number)}, cols: {segment_cols}, rows: {segment_rows}")

        rasterizer = Rasterizer()
        for i in range(segment_rows):
            for j in range(segment_cols):
                grid = rasterizer.rasterize_segment_from_indexes(gdf_edges=gdf_edges, indexes=(i, j), size_h=self.segment_h, size_w=self.segment_w, pixel_size=self.grid_density)
                print(f"Segment: {i}, {j}")
                print(f"Grid segment shape: {grid.shape}")
                grid_manager.write_segment(grid, i, j)
        return grid_manager

    def add_elevation_to_grid(self, grid_manager: GridManager):
        """
        Enriches the existing grid with elevation data retrieved from NASA SRTM.

        This method iterates through grid segments, converts local map coordinates
        (e.g., UTM EPSG:32634) to WGS84 (Lat/Lon) if necessary, queries the SRTM
        database for altitude, and updates the grid files.

        Args:
            grid_manager (GridManager): The manager object handling grid file I/O.
        """
        import srtm
        from pyproj import Transformer

        print("Initializing SRTM data provider...")
        geo_data = srtm.get_data()
        meta = grid_manager.get_metadata()

        is_metric = not (-180 <= meta.upper_left_longitude <= 180)
        transformer = None

        if is_metric:
            print(f"Metric coordinates detected (X={meta.upper_left_longitude:.2f}). "
                  f"Initializing EPSG:32634 -> EPSG:4326 transformer.")
            transformer = Transformer.from_crs("EPSG:32634", "EPSG:4326", always_xy=True)
        else:
            print("Geographic coordinates detected. No conversion required.")

        segments_rows = math.ceil(meta.rows_number / meta.segment_h)
        segments_cols = math.ceil(meta.columns_number / meta.segment_w)

        print(f"Processing elevation for {segments_rows}x{segments_cols} segments...")

        for row_idx in range(segments_rows):
            for col_idx in range(segments_cols):
                segment = grid_manager.read_segment(row_idx, col_idx)

                h, w = segment.shape[:2]

                height_map = np.zeros((h, w), dtype=np.float32)

                for y in range(h):
                    global_row = row_idx * meta.segment_h + y
                    current_y_map = meta.upper_left_latitude - (global_row * meta.grid_density)

                    for x in range(w):
                        global_col = col_idx * meta.segment_w + x
                        current_x_map = meta.upper_left_longitude + (global_col * meta.grid_density)

                        if transformer:
                            lon, lat = transformer.transform(current_x_map, current_y_map)
                        else:
                            lon, lat = current_x_map, current_y_map

                        height_map[y, x] = self._get_altitude_source(lat, lon, geo_data)

                segment[:, :, 1] = height_map
                grid_manager.write_segment(segment, row_idx, col_idx)

                print(f"Segment [{row_idx}, {col_idx}] saved. Max elevation: {np.max(height_map):.2f} m")

    def _get_altitude_source(self, lat: float, lon: float, geo_data=None) -> float:
        """
        Safely retrieves elevation from the SRTM data source.

        Args:
            lat (float): Latitude in degrees.
            lon (float): Longitude in degrees.
            geo_data: The SRTM data object.

        Returns:
            float: Elevation in meters, or 0.0 if data is missing or invalid.
        """
        if geo_data is None:
            return 0.0

        try:
            elevation = geo_data.get_elevation(lat, lon)

            if elevation is None:
                return 0.0

            return float(elevation)

        except Exception:
            return 0.0