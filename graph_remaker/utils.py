from grid_manager import GRID_INDICES, Grid
from graph_remaker.data_structures import GridPoint


def is_border_point(grid_part: Grid, point: GridPoint) -> bool:
    y, x = point
    neighborhood = grid_part[y - 1 : y + 2, x - 1 : x + 2, GRID_INDICES.IS_STREET]
    return neighborhood.sum() < neighborhood.size