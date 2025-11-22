from grid_manager import GridManager

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

    def load_city_grid(city: str, file_name: str) -> GridManager:
        """Load city grid to a given file.
        Args:
            city (str): String for identification of the city (OSM-like).
            file_name (str): Target file name.
        Returns:
            grid_manager (GridManager): Object handling partial load/write to the specified file.
        Raises:
            FileExistsError: if file with specified name already exists.
            """

