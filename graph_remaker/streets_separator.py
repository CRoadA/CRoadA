from data_manager import GRID_INDICES, Grid
from graph_remaker.data_structures import GridPoint, StreetBorder
from graph_remaker.utils import is_border_point



def _find_opposite_point(grid_part: Grid, borders: list[StreetBorder], borders_of_points: dict[GridPoint, StreetBorder], point: GridPoint, step: int) -> GridPoint:
    """Find an opposite point on the other side of the street. It can either belong to the other border or (in case of dead-end) to the same as given.

    Parameters
    ----------
    grid_part : Grid
        Grid part to find next point on.
    borders : list[StreetBorder]
        _description_
    borders_of_points : dict[GridPoint, StreetBorder]
        Dict mapping border points to their borders.
    point : GridPoint
        Relative point.
    step : int
        Number of border points traversed along the border between finding opposite points on the other border (parameter of street identification algorithm).

    Returns
    -------
    GridPoint
        Border point satisfying conditons to be 'on the other side' of the street (not necessarily on a different border).
    """

    assert is_border_point(grid_part, point), f"Given point {point} is not border point"

    currentBorder = borders_of_points[point]
    queue = [point]
    stratum_number = 0
    
    neighbors = _get_grid_neighbors(grid_part, point, [])
    if not neighbors:
        return None
    first_of_next_stratum = neighbors[0]
    visited = []

    while queue:
        checked_point = queue.pop(0)

        # get neighbors to enqueue
        neighbors = _get_grid_neighbors(grid_part, point, visited)

        # check, if new stratum started
        if checked_point == first_of_next_stratum:
            stratum_number += 1
            first_of_next_stratum = neighbors[0]

        # If satisfies conditions - return the point
        if is_border_point(grid_part, checked_point):
            # Different border case
            checked_border = borders_of_points[checked_point]
            if checked_border != currentBorder:
                return checked_point
            
            # Dead-end case
            distance_along_border = currentBorder.calculate_distance_between_points(checked_point, point)
            if distance_along_border - stratum_number > step:
                return checked_point

        visited.append(point)

        queue += neighbors
        
    
    return None     
            
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


step = 5

def identify_streets(grid_part: Grid, borders: list[StreetBorder], borders_of_points: dict[GridPoint, StreetBorder]):
    for current_border in borders:
        points = current_border.to_list()

        last_point = points[0]
        last_opposite_point = _find_opposite_point(grid_part, borders, borders_of_points, last_point, step)

        foregoing_opposite_border = borders_of_points[last_opposite_point]


        if current_border == foregoing_opposite_border:
            # TODO handle dead-end
            pass


        for point in points[step::step]:
            opposite_point = _find_opposite_point(grid_part, borders, borders_of_points, point, step)
            new_opposite_border = borders_of_points[opposite_point]

            if borders_of_points[point] == new_opposite_border:
                # TODO handle dead-end
                pass

            if new_opposite_border != foregoing_opposite_border:
                # TODO handle crossroad






        