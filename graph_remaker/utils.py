from grid_manager import GRID_INDICES, Grid
from graph_remaker.data_structures import GridPoint


def is_border_point(grid_part: Grid, point: GridPoint) -> bool:
    """Determine, if point is part of a street border.

    Parameters
    ----------
    grid_part : Grid
        Grid part to dertermine the fact.
    point : GridPoint
        Checked point.

    Returns
    -------
    bool
        True, if point is a part of some border, otherwise false.
    """
    y, x = point
    y_min = max(0, y - 1)
    y_max = max(grid_part.shape[0], y + 2)
    x_min = max(0, x - 1)
    x_max = max(grid_part.shape[1], x + 2)
    neighborhood = grid_part[y_min : y_max, x_min : x_max, GRID_INDICES.IS_STREET]
    return neighborhood.sum() < neighborhood.size

def is_conflicting_point(grid_part: Grid, point: GridPoint) -> bool:
    """Determine, if the point is conflicting (i.e. is near enough to grid part border, currently).

    Parameters
    ----------
    grid_part : Grid
        Grid part to dertermine the fact.
    point : GridPoint
        Checked point.

    Returns
    -------
    bool
        True, if the point is conflicting, otherwise false.
    """
    height, width = grid_part.shape
    return (1, 1) <= point <= (height - 2, width - 2)