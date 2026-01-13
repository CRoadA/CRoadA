from dataclasses import dataclass
from typing import Any
from integrated import MeshPoint, Grid, discover_streets
import numpy as np
import math

overlap = 1

def divide_into_parts(grid: Grid, part_size: int = 200):
    width, height = grid.shape
    x_parts_number = math.ceil(width / part_size)
    y_parts_number = math.ceil(height / part_size)

    streets: list[list[MeshPoint]] = [] # TODO linestrings?
    crossroads: list[list[int]] = [] # TODO street ids?
    for row in range(x_parts_number):
        # lower/upper in the sense of indices values
        # parts overlap by 1 row
        y_lower = max(0, row * part_size - overlap)
        y_upper = min(0, (row + 1) * part_size + overlap)
        for col in range(y_parts_number):
            # parts overlap by 1 column
            x_lower = max(0, col * part_size - overlap)
            x_upper = min(width, (col + 1) * part_size + overlap)
            street_discoveries, crossroad_discoveries = discover_streets(grid[y_lower:y_upper, x_lower:x_upper])
            # TODO resolve conflicts by merging or removing?
            # TODO streets.append()
            # TODO crossroads.append()

    # TODO return OSMnx-like format?