import numpy as np
from dataclasses import dataclass
from typing import Any
from data_manager import GRID_INDICES, Grid
from .data_structures import GridPoint, StreetBorder

MeshPoint = tuple[float, float] # point on Mesh

@dataclass
class StreetConflict:
    """Conflict of street with crossroads or grid parrt border.
    Attributes:
        conflict_points (list[GridPoint]): Grid points involved in conflict.
        linestring_points (list[MeshPoint]): Points of street's linestring involved in conflict.
    """
    conflict_points: list[GridPoint]
    linestring_points: list[MeshPoint]

@dataclass
class StreetDiscovery:
    """Discovered data about street during the street discovery process.
    Attributes:
        linestring (list[MeshPoint]): Graph representation of street (part of Mesh). May be empty.
        borders (list[StreetBorder]): Discovered borders of Street.
        conflicts (list[StreetConflict]): Conflicts with Crossroads or grid part border.
    """
    linestring: list[MeshPoint]
    borders: list[StreetBorder]
    conflicts: list[StreetConflict]



@dataclass
class CrossroadDiscovery:
    """Discovered data about street during the street discovery process.
    Attributes:
        points (list[GridPoint]): Points in the interior of the crossroad.
        conflicting_points (list[GridPoint]): Conflicting points of the interior of the crossroad.
        street_juntions (list[tuple[StreetDiscovery, MeshPoint]]): List of adjacentStreetDiscoveries with their junction points (parts of Mesh) binding the StreetDiscovery to the crossroad.
    """
    points: list[GridPoint]
    conflicting_points: list[GridPoint]
    street_juntions: list[tuple[StreetDiscovery, MeshPoint]]

    

def discover_streets(grid_part: Grid) -> tuple[list[StreetDiscovery], list[CrossroadDiscovery]]:
    """Discovers streets for grid part.
    Args:
        grid_part (Grid): searched through grid part.
    Returns:
        tuple consisting of:
            - street_discoveries (list[StreetDiscovery]): Discovered streets with their conflicts.
            - crossroad_discoveries (list[CrossroadDiscovery]): Discovered crossroads with their conflicts.
    """
    pass

def is_border_point(grid_part: Grid, point: GridPoint):
    y, x = point
    neighborhood = grid_part[y - 1 : y + 2, x - 1 : x + 2, GRID_INDICES.IS_STREET]
    return neighborhood.sum() < neighborhood.size

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
    is_found, (row, col) = find_first_non_checked_street(grid_part)
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

def _find_opposite_border(grid_part: Grid, borders: list[StreetBorder], borders_of_points: dict[GridPoint, StreetBorder], point: GridPoint):
    pass

step = 5

def identify_streets(grid_part: Grid, borders: list[StreetBorder], borders_of_points: dict[GridPoint, StreetBorder]):
    for border in borders:
        points = border.to_list()
        for point in points[::step]:
            #TODO
            # _find_opposite_border()
            pass