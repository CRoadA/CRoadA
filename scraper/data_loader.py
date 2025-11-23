from grid_manager import GridManager
from scraper.grid_builder import GridBuilder 
from scraper.rasterizer import Rasterizer
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

