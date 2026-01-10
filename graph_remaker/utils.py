from collections.abc import Callable
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
    row, col = point
    return (
        row == 0 or row == height - 1
        or col == 0 or col == width - 1
    )

def _get_grid_neighbors(grid_part: Grid, point: GridPoint, visited: list[GridPoint]) -> list[GridPoint]:
    """Get neighbor points.

    Parameters
    ----------
    grid_part : Grid
        Grid to find neighbors on.
    point : GridPoint
        Point, which neighbors need to be found.
    visited : list[GridPoint]
        Already visited points.

    Returns
    -------
    list[GridPoint]
        List of neighbors of the point (von Neumann neighborhood).
    """
    y, x = point
    neighbors = []

    checked_coords = [
        (y-1, x),
        (y+1, x),
        (y, x-1),
        (y, x+1)
    ]
    for n_y, n_x in checked_coords:
        if (n_y, n_x) not in visited and grid_part[n_y, n_x, GRID_INDICES.IS_STREET]:
            neighbors.append((n_y, n_x))
            
    return neighbors

def bfs(grid_part: Grid, handle: GridPoint, predicate: Callable[[GridPoint], bool]):
    """Pursue Breadth-First Search with specified parameters.

    Parameters
    ----------
    grid_part : Grid
        Grid to pursue BFS on.
    handle : GridPoint
        Starting point.
    predicate : Callable[[GridPoint], bool]
        Function of cosidered point. If it returns True, its neighbors are going to be processed later.
    """
    queue = [handle]
    visited = []

    while queue:
        point = queue.pop(0)
        visited.append(point)
        if not predicate(point):
            continue

        queue.extend(_get_grid_neighbors(grid_part, point, visited))