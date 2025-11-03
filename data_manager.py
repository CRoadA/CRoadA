import pickle
import os
import numpy as np
from typing import Any
from dataclasses import dataclass
        
IS_STREET_INDEX = 0
ALTITUDE_INDEX = 1

    


def write(grid: np.ndarray[(Any, Any, Any), Any], file_name: str, data_dir: str = "grids"):
    assert grid.shape[2] == 2
    write_pkl(grid, file_name, data_dir)

def read(file_name: str, data_dir: str = "grids"):
    return read_pkl(file_name, data_dir)

# specific methods - better use general-purpose

def write_pkl(grid: np.ndarray[(Any, Any, Any), Any], file_name: str, data_dir: str = "grids"):
    with open(os.path.join(data_dir, file_name), "wb") as file:
        pickle.dump(grid, file)

def read_pkl(file_name: str, data_dir: str = "grids"):
    with open(os.path.join(data_dir, file_name), "rb") as file:
        return pickle.load(file)

# Appears to be useless    

@dataclass
class Coordinates:
    x: int
    y: int

@dataclass
class Point:
    y: int
    x: int
    is_street: bool
    altitude: float

class Grid:

    _original_matrix: np.ndarray[(Any, Any, Any), Any]
    _matrix: np.ndarray[(Any, Any, Any), Point]

    # helper functions
    _are_streets = np.vectorize(lambda point: point.is_street)
    _get_altitudes = np.vectorize(lambda point: point.altitude)
    
    def __init__(self, grid: np.ndarray[(Any, Any, Any), Any]):
        self._original_matrix = grid

        X, Y, _ = grid.shape
        self._matrix = np.zeros((X, Y))

        for row in range(X):
            for col in range(Y):
                values = grid[row, col]
                self._matrix[row, col] = Point(
                    row, col,
                    values[0] == 1,
                    values[1]
                )

    def __getitem__(self, index):
        return self._matrix[index]

    """
        Get matrix stating if the point is part of the street for whole
        Grid.
    """
    def get_are_streets(self):
        return self._are_streets(self._original_matrix)
    
    """
        Get matrix with altitiudes of points all across the Grid.
    """
    def get_altitudes(self):
        return self._get_altitudes(self._original_matrix)