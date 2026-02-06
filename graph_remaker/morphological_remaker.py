import numpy as np
from dataclasses import dataclass
from typing import Any
from grid_manager import GRID_INDICES, Grid
from .data_structures import CrossroadDiscovery, GridPoint, StreetDiscovery
import skimage.morphology as skim
from scipy.ndimage import label
from .utils import is_conflicting_point, bfs

import matplotlib.pyplot as plt


def _enrich_street_metrics(streets: list[StreetDiscovery], grid_part: Grid, density: float):
    """
    Oblicza szerokość, nachylenie podłużne i poprzeczne dla każdej drogi.
    """
    is_street_mask = grid_part[:, :, GRID_INDICES.IS_STREET]
    altitudes = grid_part[:, :, GRID_INDICES.ALTITUDE]

    for street in streets:
        pts = street.linestring
        if len(pts) < 2:
            continue

        long_slopes = []
        widths = []
        trans_slopes = []

        # Próbkujemy co kilka punktów
        step = 3
        for i in range(0, len(pts) - 1, step):
            p1 = pts[i]
            p2 = pts[min(i + step, len(pts) - 1)]

            # 1. NACHYLENIE PODŁUŻNE
            h1 = altitudes[p1[0], p1[1]]
            h2 = altitudes[p2[0], p2[1]]
            dist_px = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            dist_m = dist_px * density

            if dist_m > 0:
                slope = abs(h2 - h1) / dist_m
                long_slopes.append(slope)


            dy = p2[0] - p1[0]
            dx = p2[1] - p1[1]

            # Wektor prostopadły (normalna)
            norm_y, norm_x = -dx, dy
            length = np.sqrt(norm_y ** 2 + norm_x ** 2)
            if length == 0: continue

            unit_y = norm_y / length
            unit_x = norm_x / length

            # Szukamy krawędzi drogi w obie strony (lewo/prawo)
            edge_l, h_l = _find_road_edge(p1, (unit_y, unit_x), is_street_mask, altitudes)
            edge_r, h_r = _find_road_edge(p1, (-unit_y, -unit_x), is_street_mask, altitudes)

            width_m = (edge_l + edge_r) * density
            if width_m > 0:
                widths.append(width_m)
                # Nachylenie poprzeczne:
                t_slope = abs(h_l - h_r) / width_m
                trans_slopes.append(t_slope)

        street.max_longitudinal_slope = max(long_slopes) if long_slopes else 0.0
        street.max_transversal_slope = max(trans_slopes) if trans_slopes else 0.0
        street.width = np.median(widths) if widths else 0.0


def _find_road_edge(start_pt, direction, mask, altitudes):
    """Kroczy wzdłuż wektora aż napotka koniec maski drogi."""
    curr_y, curr_x = float(start_pt[0]), float(start_pt[1])
    h, w = mask.shape
    dist = 0
    last_h = altitudes[start_pt[0], start_pt[1]]

    while True:
        next_y = int(round(curr_y + direction[0]))
        next_x = int(round(curr_x + direction[1]))

        # Wyjście poza macierz lub poza maskę drogi
        if next_y < 0 or next_y >= h or next_x < 0 or next_x >= w:
            break
        if not mask[next_y, next_x]:
            break

        curr_y += direction[0]
        curr_x += direction[1]
        dist += 1
        last_h = altitudes[next_y, next_x]

        if dist > 100: break  # Safety break (droga nie może mieć 1km szerokości)

    return dist, last_h

def discover_streets(grid_part: Grid) -> tuple[list[CrossroadDiscovery], list[CrossroadDiscovery], list[StreetDiscovery], list[StreetDiscovery]]:
    """Discover streets for given grid part.

    Parameters
    ----------
    grid_part : Grid
        Grid part, on which crossroads and streets need to be discovered.

    Returns
    -------
    conflictless_crossroads: list[CrossroadDiscovery]
        List of crossroads without conflicts (not touching grid part border).
    conflicting_crossroads: list[CrossroadDiscovery]
        List of crossroads with conflicts (touching grid part border).
    conflictless_streets: list[StreetDiscoveries]
        List of streets without conflicts (not touching grid part border).
    conflicting_streets: list[StreetDiscoveries]
        List of streets with conflicts (touching grid part border).
    """
    SKELET_B = 5 # skeleton border

    is_street_image = grid_part[:, :, GRID_INDICES.IS_STREET]

    bigger_shape = (is_street_image.shape[0] + 2 * SKELET_B, is_street_image.shape[1] + 2 * SKELET_B)
    bigger_for_skeletonization = np.ones(bigger_shape)
    bigger_for_skeletonization[SKELET_B: bigger_shape[0]-SKELET_B, SKELET_B: bigger_shape[1]-SKELET_B] = is_street_image
    skeletonized = skim.skeletonize(bigger_for_skeletonization)
    skeletonized = skeletonized[SKELET_B: bigger_shape[0]-SKELET_B, SKELET_B: bigger_shape[1]-SKELET_B]

    crossroads_image = _find_crossroads(skeletonized)
    separated_streets_image = np.logical_and(skeletonized, np.logical_not(crossroads_image))

    border = np.zeros(is_street_image.shape)
    border[0, :] = 1
    border[-1, :] = 1
    border[:, 0] = 1
    border[:, -1] = 1
    seed = np.logical_and(border, crossroads_image)

    conflicting_crossroads_image = skim.reconstruction(seed, crossroads_image)
    conflictless_crossroads_image = np.logical_and(crossroads_image, np.logical_not(conflicting_crossroads_image))

    conflicting_crossroads_by_points, conflicting_crossroads = _create_crossroads(conflicting_crossroads_image)
    conflictless_crossroads_by_points, conflictless_crossroads = _create_crossroads(conflictless_crossroads_image)

    seed = np.logical_and(border, separated_streets_image)
    conflicting_streets_image = skim.reconstruction(seed, separated_streets_image)
    conflictless_streets_image = np.logical_and(separated_streets_image, np.logical_not(conflicting_streets_image))

    conflicting_streets_by_points, conflicting_streets = _create_street_dicoveries(conflicting_streets_image)
    conflictless_streets_by_points, conflictless_streets = _create_street_dicoveries(conflictless_streets_image)

    street_starts, street_ends = _assign_streets_to_crossroads(
        conflicting_crossroads + conflictless_crossroads,
        conflicting_streets_by_points | conflictless_streets_by_points
    )


    _supply_street_ends_topologically(
        separated_streets_image,
        conflicting_streets + conflictless_streets,
        street_starts,
        street_ends
        )

    
    _order_street_linestrings(
        separated_streets_image,
        conflicting_streets + conflictless_streets,
        street_starts,
        street_ends
    )
    density = 1.0  # właściwą wartość z GridManager
    _enrich_street_metrics(conflicting_streets + conflictless_streets, grid_part, density)

    return (
        conflictless_crossroads,
        conflicting_crossroads,
        conflictless_streets,
        conflicting_streets,
    )

def _assign_streets_to_crossroads(crossroads: list[CrossroadDiscovery], streets_by_points: dict[GridPoint, StreetDiscovery]) -> tuple[dict[StreetDiscovery, GridPoint], dict[StreetDiscovery, GridPoint]]:
    """Update crossroads discoveries with adjacent streets. During the process dictionaries of streets' starts and ends are created.

    Parameters
    ----------
    crossroads : list[CrossroadDiscovery]
        List of crossroads discoveries, which need to be supplied.
    streets_by_points : dict[GridPoint, StreetDiscovery]
        Street discoveries by points which belong to them.

    Returns
    -------
    street_starts: dict[StreetDiscovery, GridPoint]
        Start of linestring for each street discovery.
    street_ends: dict[StreetDiscovery, GridPoint]
        End of linestring for each street discovery.
    """

    street_starts: dict[StreetDiscovery, GridPoint] = dict()
    street_ends: dict[StreetDiscovery, GridPoint] = dict()

    for crossroad in crossroads:
        for row, column in crossroad.points:
            for i, j in np.ndindex(3, 3):
                if i == 1 and j == 1:
                    continue

                checkedRow, checkedColumn = row + i - 1, column + j - 1
                if (checkedRow, checkedColumn) in streets_by_points: # this is not satisfied for points exceeding grid part
                    point = (checkedRow, checkedColumn)
                    street = streets_by_points[point]
                    
                    if street in crossroad.street_junctions:
                        continue

                    crossroad.street_junctions[street] = point

                    if street in street_starts:
                        street_ends[street] = point
                    else:
                        street_starts[street] = point

    return street_starts, street_ends

def _supply_street_ends_topologically(separated_streets_image: np.ndarray[(Any, Any), bool], streets: list[StreetDiscovery], street_starts: dict[StreetDiscovery, GridPoint], street_ends: dict[StreetDiscovery, GridPoint]):

    counter = 0
    for street in streets:
        if street not in street_starts:
            start, end = _find_street_topological_ends(separated_streets_image, street)
            street_starts[street] = start
            street_ends[street] = end
        elif street not in street_ends:
            _, end = _find_street_topological_ends(separated_streets_image, street, start_point=street_starts[street])
            street_ends[street] = end
        counter += 1

BFSGridNode = tuple[GridPoint, GridPoint] # origin and actual point

def _find_street_topological_ends(separated_streets_image: np.ndarray[(Any, Any), np.bool], street: StreetDiscovery, start_point: GridPoint | None = None):

    def _queue_neighbors_if_necessary(node: GridPoint, separated_streets_image: np.ndarray[(Any, Any), np.bool], queue: list[BFSGridNode]):

        _, point = node
        y, x = point
        #w, h = separated_streets_image.shape
        h, w = separated_streets_image.shape

        checked_coords = [
            (max(0, y-1), max(0, x-1)),
            (max(0, y-1), x),
            (max(0, y-1), min(w - 1, x+1)),
            (y, max(0, x-1)),
            (y, x),
            (y, min(w - 1, x+1)),
            (min(h - 1, y+1), max(0, x-1)),
            (min(h - 1, y+1), x),
            (min(h - 1, y+1), min(w - 1, x+1)),
        ]
        for n_y, n_x in checked_coords:
            if separated_streets_image[n_y, n_x]:
                queue.append((point, (n_y, n_x)))


    # random linestring point
    handle = street.linestring[0] if start_point is None else start_point

    queue: list[BFSGridNode] = [(None, handle)] # (origin, point to visit)
    visited: list[GridPoint] = []
    paths: list[list[GridPoint]] = []
    
    counter = 0
    while queue:
        node = queue.pop(0)
        origin, point = node
        if point in visited:
            continue
        visited.append(point)

        appropriate_path = next((path for path in paths if origin in path), None)
        if appropriate_path is None:
            # it means this is the start point
            paths.append([handle])
        else:
            if appropriate_path[-1] == origin:
                appropriate_path.append(point)
            else:
                index = appropriate_path.index(origin)
                new_path = appropriate_path[:index + 1]
                new_path.append(point)
                paths.append(new_path)

        _queue_neighbors_if_necessary(node, separated_streets_image, queue)
        counter += 1

    # Find longest path
    first_path = paths[0]
    for path in paths:
        if len(first_path) < len(path):
            first_path = path
    # Find longest path disjoint with first
    second_path = []
    for path in paths:
        # below is used index == 1, because at index == 0 all paths have the same point (handle)
        if len(path) > 1 and len(first_path) > 1:
            if path[1] != first_path[1] and len(path) > len(second_path):
                second_path = path
    if not second_path: # if no disjoint path was found
        second_path = [first_path[0]] # use the handle as the opposite end

    return first_path[-1], second_path[-1]

def _order_street_linestrings(separated_streets_image: np.ndarray[(Any, Any), np.bool], streets: list[StreetDiscovery], street_starts: dict[StreetDiscovery, GridPoint], street_ends: dict[StreetDiscovery, GridPoint]):

    def _queue_neighbors_if_necessary(point: GridPoint, street: StreetDiscovery, separated_streets_image: np.ndarray[(Any, Any), np.bool], queue: list[GridPoint]):

        y, x = point
        # w, h = separated_streets_image.shape
        h, w = separated_streets_image.shape
        queued = []

        checked_coords = [
            (max(0, y-1), max(0, x-1)),
            (max(0, y-1), x),
            (max(0, y-1), min(w - 1, x+1)),
            (y, max(0, x-1)),
            (y, x),
            (y, min(w - 1, x+1)),
            (min(h - 1, y+1), max(0, x-1)),
            (min(h - 1, y+1), x),
            (min(h - 1, y+1), min(w - 1, x+1)),
        ]
        for n_y, n_x in checked_coords:
            if separated_streets_image[n_y, n_x] and (n_y, n_x) in street.linestring:
                queue.append((n_y, n_x))
                queued.append((n_y, n_x))
        return queued

    for street in streets:
        new_linestring = []
        start = street_starts[street]
        end = street_ends[street]

        queue: list[GridPoint] = [start]
        visited: list[GridPoint] = []
        while queue:
            point = queue.pop(0)
            if point in visited:
                continue
            visited.append(point)

            if point != end : # end point needs to be added by hand later
                new_linestring.append(point)
            
            _queue_neighbors_if_necessary(point, street, separated_streets_image, queue)

        new_linestring.append(end)
        street.linestring = new_linestring                    

def _find_crossroads(skeleton: np.ndarray[(Any, Any), np.bool]):
    RING_LENGTH = 8

    height, width = skeleton.shape
    result = np.zeros(skeleton.shape)

    for row in range(1, height - 1):
        for column in range(1, width - 1):
            neighborhood = skeleton[row - 1: row + 2, column - 1 : column + 2]
            flat = neighborhood.flatten()
            listed = [*(flat[0:3]), flat[5], *(reversed(flat[6:9])), flat[3]]

            # Spin the ring in a way, that none of the 1s' regions is separated via list beginning/end
            # Find first zero
            start_index = -1
            try:
                start_index = listed.index(0)
            except ValueError:
                continue

            ring = [*(listed[start_index:]), *(listed[:start_index])]

            # Check, if 3 separated 1s' regions
            # First 0s region

            counter = _find_value_region_end(ring, 0)
            if counter == RING_LENGTH:
                continue
            # First 1s region
            counter = _find_value_region_end(ring, counter)
            if counter == RING_LENGTH:
                continue
            # Second 0s region
            counter = _find_value_region_end(ring, counter)
            if counter == RING_LENGTH:
                continue
            # Second 1s region
            counter = _find_value_region_end(ring, counter)
            if counter == RING_LENGTH:
                continue
            # Third 0s region
            counter = _find_value_region_end(ring, counter)
            if counter == RING_LENGTH:
                continue

            result[row - 1: row + 2, column - 1 : column + 2] = 1

    return result

def _create_crossroads(crossroads_image: np.ndarray[(Any, Any), bool]) -> tuple[dict[GridPoint, CrossroadDiscovery], CrossroadDiscovery]:
    
    structure = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])

    labeled_image, num_labels = label(crossroads_image, structure)
    crossroad_discoveries: list[CrossroadDiscovery] = []
    result = dict()

    for i in range(1, num_labels + 1):
        crossroad_discoveries.append(CrossroadDiscovery())
    
    for (row, column), crossroad_label in np.ndenumerate(labeled_image):
        if crossroad_label == 0:
            continue

        crossroad = crossroad_discoveries[crossroad_label - 1]
        added_point = (row, column)
        crossroad.points.append(added_point)
        if is_conflicting_point(crossroads_image, added_point):
            crossroad.conflicting_points.append(added_point)

        result[added_point] = crossroad

    return result, crossroad_discoveries


def _create_street_dicoveries(separated_roads_image: np.ndarray[(Any, Any), bool]) -> tuple[dict[GridPoint, StreetDiscovery], list[StreetDiscovery]]:
    """Create street discoveries dictionary from separated streets image. Returned streets DO NOT contain real linestrings, but only an unordered points list for now. User must order them propely self.
    """

    structure = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    labeled_image, num_labels = label(separated_roads_image, structure)
    street_discoveries: StreetDiscovery = []
    result = dict()

    for i in range(1, num_labels + 1):
        street_discoveries.append(StreetDiscovery())
    
    for (row, column), street_label in np.ndenumerate(labeled_image):
        if street_label == 0:
            continue

        street = street_discoveries[street_label - 1]
        added_point = (row, column)
        street.linestring.append(added_point)
        if is_conflicting_point(separated_roads_image, added_point):
            street.conflicts.append(added_point)

        result[added_point] = street

    return result, street_discoveries

def _find_value_region_end(array: list, start_index: Any) -> int:
    """Find the first occurence of different value starting from index.

    Parameters
    ----------
    array : list
        Searched through list.
    counter : Any
        Starting index.

    Returns
    -------
    int
        Index of first occurence of different value. Array length if not found.
    """
    compared = array[start_index]
    for index in range(start_index + 1, len(array)):
        value = array[index]
        if compared != value:
            return index
    return len(array)
        
