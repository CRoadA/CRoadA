from grid_manager import GRID_INDICES, Grid
from graph_remaker.data_structures import GridPoint, StreetBorder, StreetDiscovery, CrossroadDiscovery
from graph_remaker.utils import is_border_point, is_conflicting_point
from collections.abc import Callable



def _find_opposite_point(grid_part: Grid, borders_of_points: dict[GridPoint, StreetBorder], point: GridPoint, step: int) -> GridPoint | None:
    """Find an opposite point on the other side of the street. It can either belong to the other border or to the same as given (in case of dead-end).

    Parameters
    ----------
    grid_part : Grid
        Grid part to find next point on.
    borders_of_points : dict[GridPoint, StreetBorder]
        Dict mapping border points to their borders.
    point : GridPoint
        Relative point.
    step : int
        Number of border points traversed along the border between finding opposite points on the other border (parameter of street identification algorithm).

    Returns
    -------
    opposite_point: GridPoint
        Border point satisfying conditons to be 'on the other side' of the street (not necessarily on a different border), or None, if no such found.
    """
    currentBorder = borders_of_points[point]

    def predicate(checked_point: GridPoint, stratum_number: int) -> bool:
        # Different border case
        checked_border = borders_of_points[checked_point]
        if checked_border != currentBorder:
            return True
        
        # Dead-end case
        distance_along_border = currentBorder.calculate_distance_between_points(checked_point, point)
        if distance_along_border - stratum_number > step:
            return True
        
        return False        
    
    return _find_opposite_point_satisfying_condition(
        grid_part,
        point,
        predicate
        )

def _find_opposite_point_on_border(grid_part: Grid, borders_of_points: dict[GridPoint, StreetBorder], point: GridPoint, border: StreetBorder) -> GridPoint | None:
    """Find the point on the opposite side of the street, but necessarily on specified border.

    Parameters
    ----------
    grid_part : Grid
        Grid part to find the point on.
    borders_of_points : dict[GridPoint, StreetBorder]
        Dict mapping border points to their borders.
    point : GridPoint
        Relative point.
    border : StreetBorder
        Border, on which the searched point needs to be.

    Returns
    -------
    GridPoint
        Nearest to the relative point (in terms of Breadth-First Search) point on the given border or None, if no such found.
    """
    def predicate(checked_point, _):
        checked_point_border = borders_of_points[checked_point]
        if checked_point_border == border:
            return True
        
    return _find_opposite_point_satisfying_condition(
        grid_part,
        point,
        predicate
        )

def _find_opposite_point_satisfying_condition(grid_part: Grid, point: GridPoint, predicate: Callable[[GridPoint, int], bool]) -> GridPoint | None:
    """Find first point satisfying conditions with Breadth-First Search.

    Parameters
    ----------
    grid_part : Grid
        Grid part to find the point on.
    point : GridPoint
        Relative point.
    predicate: Callable[[GridPoint, int], bool]
        Predicate, indicating, if the point satifies conditions. Its argumets are:
            * checked_point (GridPoint): Regarded point.
            * stratum_number (int): Ordinal number of the (Breadth-First Search) stratum of the point.

    Returns
    -------
    GridPoint
        Nearest (in terms of Breadth-First Search) border point satisfying conditon or None, if no such found.
    """

    assert is_border_point(grid_part, point), f"Given point {point} is not border point"

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
            if predicate(checked_point, stratum_number):
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


# step = 5

def _calc_grid_manhattan_distance(first: GridPoint, second: GridPoint):
    dx = first[0] - second[0]
    dy = first[1] - second[1]
    return abs(dx) + abs(dy)

def _discover_cohesive_conflicts_area(grid_part: Grid, handle: GridPoint, stop: GridPoint = None) -> list[GridPoint]:
    """Discover cohesive conflicts area with Breadth-First Search.
    """
    conflicts = []

    def callback(point: GridPoint) -> bool:
        if not is_conflicting_point(grid_part, point):
            return True
        if not grid_part[point[0], point[1], GRID_INDICES.IS_STREET]:
            return True
        
        conflicts.append(point)

        if point == stop:
            return True
        return False

    _bfs(grid_part, handle, callback)
    return callback

# def _find_touching_grid_part_borders_street_border_ends(grid_part: Grid, border: StreetBorder):
#     """Find points of border, which lie on a grid part border (i.e. conflicting ones). It is not guaranteed by e.g. StreetBorder.to_list()[0].
#     """
#     first

def _bfs(grid_part: Grid, handle: GridPoint, predicate: Callable[[GridPoint], bool]):
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



def identify_streets(grid_part: Grid, borders: list[StreetBorder], borders_of_points: dict[GridPoint, StreetBorder], step: int = 5):

    # TODO find conflicts in discovered streets
    discovered_streets: list[StreetDiscovery] = []
    borders_queue: list[StreetBorder] = borders.copy()

    while borders_queue:
        foregoing_border = borders_queue.pop(0)
        foregoing_border_points = foregoing_border.to_list()
        foregoing_point = foregoing_border_points[0]

        foregoing_opposite_point = _find_opposite_point(grid_part, borders_of_points, foregoing_point, step)

        if foregoing_opposite_point == None:
            # It has to be a single border with no dead-ends with conflicting points
            # for conflicts it needs to be assumed, that there are multiple "adjacency points" of the street border with the grid part border.
            street = StreetDiscovery(
                [],
                [foregoing_border],
                _discover_cohesive_conflicts_area(
                    grid_part,
                    foregoing_border_points[0], # TODO unfortunatelly cannot assume that
                    foregoing_border_points[-1] # TODO unfortunatelly cannot assume that
                    )
                )


    foregoing_opposite_border = borders_of_points[foregoing_opposite_point]
    if foregoing_opposite_border == foregoing_border:
        # TODO handle dead-end
        # need to also get the real opposite point
        pass

    # first_street = StreetDiscovery([], [foregoing_border, foregoing_opposite_border], [])
    # streets_to_discover.append(first_street)

    # while streets_to_discover:
    #     street = streets_to_discover.pop(0)

    #     current_border, foregoing_opposite_border = street.borders
    #     foregoing_point = current_border[0]
    #     foregoing_opposite_point = foregoing_opposite_border[0]

    #     current_point = current_border[step] # cannot be on an opposite side of dead-end, cause it's derived from step ALONG the border
    #     current_opposite_point = _find_opposite_point(grid_part, borders_of_points, current_point, step)
    #     current_opposite_border = borders_of_points[current_opposite_point]

    #     if current_opposite_border == current_border: # If a dead-end has been met
            
    #         pass

    #     if current_opposite_border != foregoing_opposite_border: # If a new border has been met
    #         # TODO handle cross road
    #         pass


    # # Check, if no dead-end passed
    # mahattan_distance = _calc_grid_manhattan_distance(opposite_point, foregoing_opposite_point)
    # # border_distance = 


    # for foregoing_border in borders:
    #     points = foregoing_border.to_list()

    #     last_point = points[0]
    #     last_opposite_point = _find_opposite_point(grid_part, borders_of_points, last_point, step)

    #     foregoing_opposite_border = borders_of_points[last_opposite_point]


    #     if foregoing_border == foregoing_opposite_border:
    #         # TODO handle dead-end
    #         pass


    #     for point in points[step::step]:
    #         opposite_point = _find_opposite_point(grid_part, borders_of_points, point, step)
    #         new_opposite_border = borders_of_points[opposite_point]

    #         if borders_of_points[point] == new_opposite_border:
    #             # TODO handle dead-end
    #             pass

    #         if new_opposite_border != foregoing_opposite_border:
    #             # TODO handle crossroad
    #             pass

