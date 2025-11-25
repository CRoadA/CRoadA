from grid_manager import GridManager
from scraper.grid_builder import GridBuilder 
from scraper.rasterizer import Rasterizer
import numpy as np
import math
import os

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
        """Uzupełnia istniejący grid o dane wysokościowe."""

        meta = grid_manager.get_metadata()


        segments_rows = math.ceil(meta.rows_number / meta.segment_h)
        segments_cols = math.ceil(meta.columns_number / meta.segment_w)

        # Stała przybliżona: ile metrów ma jeden stopień szerokości geograficznej
        METERS_PER_DEG_LAT = 111132.0

        print(f"Rozpoczynam dodawanie wysokości dla {segments_rows}x{segments_cols} segmentów...")

        for i in range(segments_rows):
            for j in range(segments_cols):

                segment = grid_manager.read_segment(i, j)
                h, w, _ = segment.shape

                height_map = np.zeros((h, w), dtype=np.float32)

                for y in range(h):
                    global_row = i * meta.segment_h + y

                    offset_lat_m = global_row * meta.grid_density
                    current_lat = meta.upper_left_latitude - (offset_lat_m / METERS_PER_DEG_LAT)

                    meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(current_lat))

                    for x in range(w):
                        global_col = j * meta.segment_w + x

                        offset_lon_m = global_col * meta.grid_density
                        current_lon = meta.upper_left_longitude + (offset_lon_m / meters_per_deg_lon)



                        # Tutaj funkcja pobieracja wysokosc

                        altitude = self._get_altitude_source(current_lat, current_lon)
                        height_map[y, x] = altitude

                segment[:, :, 1] = height_map
                grid_manager.write_segment(segment, i, j)

                print(f"Zapisano wysokość: segment [{i}, {j}]")

    def _get_altitude_source(self, lat: float, lon: float) -> float:
        # Tymczasowa funkcja tworzaca teren
        base_height = 200.0
        wave1 = math.sin(lat * 1000) * 20
        wave2 = math.cos(lon * 1000) * 20

        return base_height + wave1 + wave2
