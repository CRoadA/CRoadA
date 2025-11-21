from data_manager import GRID_INDICES, Grid
from graph_remaker.data_structures import GridPoint, StreetBorder
from graph_remaker.utils import is_border_point
import numpy as np
from typing import Any




BFSGridNode = tuple[GridPoint, GridPoint] # origin and actual point


def _queue_neighbors_if_necessary(node: BFSGridNode, grid_part, queue: list[BFSGridNode], visited):

    _, point = node
    y, x = point

    checked_coords = [
        (y-1, x),
        (y+1, x),
        (y, x-1),
        (y, x+1)
    ]
    for n_y, n_x in checked_coords:
        # queued are points which are:
        # * normal street points, if they have not been already visited
        # * border points (always), since they may be reached from different
        # borders what means, that these borders need to be merged
        if grid_part[y, x, GRID_INDICES.IS_STREET] == 1 and (n_y, n_x) not in visited:
            queue.append((point, (n_y, n_x)))
        elif is_border_point(grid_part, (n_y, n_x)):
            queue.append((point, (n_y, n_x)))


def discover_street_borders(grid_part: Grid, handle: GridPoint):
    queue: list[BFSGridNode] = [(None, handle)] # (origin, point to visit)
    visited: list[GridPoint] = []
    borders: list[StreetBorder] = []
    borders_of_points: dict[GridPoint, StreetBorder] = {}
    
    while queue:
        node = queue.pop(0)
        origin, point = node

        # check, if borders need to be merged
        if point not in visited:
            # handle typical case
            visited.append(point)
            if is_border_point(grid_part, point):
                if origin in borders_of_points.keys():
                    border = borders_of_points[origin]
                    border.appendChild(origin, point)
                    borders_of_points[point] = border
                else: # new border part
                    new_border = StreetBorder(origin)
                    borders.append(new_border)
                    borders_of_points[point] = new_border
        else: 
            current_border = borders_of_points[point]
            removed_border = borders_of_points[origin]
            if current_border != removed_border:
                removed_border.appendChild(origin, point) # insert merging point
                current_border.merge(removed_border, point, inplace=True)
                # rebind points to new border after merging
                for p in borders_of_points.keys():
                    if borders_of_points[p] == removed_border:
                        borders_of_points[p] = current_border
                # remove deleted
                borders.remove(removed_border)
            

        _queue_neighbors_if_necessary(node, grid_part, queue, visited)
    return borders, borders_of_points, visited
    

def find_first_non_checked_street(grid: Grid, are_checked: np.ndarray[(Any, Any), bool]) -> tuple[bool, tuple[int, int]]:

    for (y, x), value in np.ndenumerate(are_checked):
        if value == 0:
            are_checked[y, x] = 1
            if grid[y, x, GRID_INDICES.IS_STREET] == True:
                return True, (x, y), value
    return False, None

def identify_borders(grid_part: Grid):
    are_checked = np.zeros(grid_part, dtype=bool)
    is_found, (row, col) = find_first_non_checked_street(grid_part, are_checked)
    borders: list[StreetBorder] = []
    borders_of_points: dict[GridPoint, StreetBorder] = {}
    while is_found:
        new_borders, new_borders_of_points, visited = discover_street_borders(grid_part, (row, col))
        borders += new_borders
        borders_of_points.update(new_borders_of_points)
        # mark all visited fields 
        for checked_y, checked_x in visited:
            are_checked[checked_y, checked_x] = 1
        # find next
        is_found, (row, col) = find_first_non_checked_street(grid_part)
    return borders, borders_of_points