import numpy as np
from dataclasses import dataclass
from typing import Any
from grid_manager import GRID_INDICES, Grid
from .data_structures import CrossroadDiscovery, GridPoint, StreetBorder, StreetDiscovery




    

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



