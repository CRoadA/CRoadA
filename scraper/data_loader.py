from grid_manager import GridManager
try:
    from scraper.grid_builder import GridBuilder
    from scraper.rasterizer import Rasterizer
except ImportError:
    pass
import numpy as np
import math
import os
import srtm
from pyproj import Transformer
import srtm
from pyproj import Transformer

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

        columns_number = math.ceil((max_x - min_x) / self.grid_density)
        rows_number = math.ceil((max_y - min_y) / self.grid_density)

        segment_rows = math.ceil((rows_number) / self.segment_h)
        segment_cols = math.ceil((columns_number) / self.segment_w)

        grid_manager = GridManager(file_name, rows_number=int(rows_number), columns_number=int(columns_number),
                                   grid_density=self.grid_density, segment_h=self.segment_h, segment_w=self.segment_w,
                                   data_dir=self.data_dir, upper_left_longitude=min_x, upper_left_latitude=max_y)
        print(f"Height: {int(rows_number)}, Width: {int(columns_number)}, rows: {segment_rows}, cols: {segment_cols}")

        rasterizer = Rasterizer()
        for i in range(segment_rows):
            for j in range(segment_cols):
                grid_2d = rasterizer.rasterize_segment_from_indexes(gdf_edges=gdf_edges, indexes=(i, j),
                                                                    size_h=self.segment_h, size_w=self.segment_w,
                                                                    pixel_size=self.grid_density)

                expected_h = self.segment_h
                if i == segment_rows - 1:
                    expected_h = rows_number % self.segment_h or self.segment_h

                expected_w = self.segment_w
                if j == segment_cols - 1:
                    expected_w = columns_number % self.segment_w or self.segment_w

                grid_3d = np.zeros((expected_h, expected_w, 2), dtype=np.float32)

                src_h, src_w = grid_2d.shape
                copy_h = min(src_h, expected_h)
                copy_w = min(src_w, expected_w)

                grid_3d[0:copy_h, 0:copy_w, 0] = grid_2d[0:copy_h, 0:copy_w]

                print(
                    f"Segment: {i}, {j} -> Expected: {expected_h}x{expected_w}, Got: {src_h}x{src_w}, Saved: {grid_3d.shape}")
                grid_manager.write_segment(grid_3d, i, j)

        return grid_manager

    def add_elevation_to_grid(self, grid_manager: GridManager):
        """
        Enriches the existing grid with elevation data retrieved from NASA SRTM.
        """

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

                h, w, _ = segment.shape

                start_lat = meta.upper_left_latitude - (row_idx * meta.segment_h * meta.grid_density)
                start_lon = meta.upper_left_longitude + (col_idx * meta.segment_w * meta.grid_density)

                for y in range(h):
                    current_y_map = start_lat - (y * meta.grid_density)

                    for x in range(w):
                        current_x_map = start_lon + (x * meta.grid_density)

                        if transformer:
                            lon, lat = transformer.transform(current_x_map, current_y_map)
                        else:
                            lon, lat = current_x_map, current_y_map

                        segment[y, x, 1] = self._get_altitude_source(lat, lon, geo_data)

                grid_manager.write_segment(segment, row_idx, col_idx)

                print(f"Segment [{row_idx}, {col_idx}] saved. Max elevation: {np.max(segment[:, :, 1]):.2f} m")

    def _get_altitude_source(self, lat: float, lon: float, geo_data=None) -> float:
        """
        Safely retrieves elevation from the SRTM data source.
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