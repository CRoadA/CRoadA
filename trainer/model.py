from abc import ABC

import numpy as np
from typing import Any
from grid_manager import GRID_INDICES, GridManager
from enum import Enum

InputGrid = np.ndarray[(Any, Any, 3), np.float64]
"""Like normal Grid, but with bools indicating, if it should be changed (the 0-th coordinate of the thrid dimension). If it is False, then the IS_STREET bool is treated as zero."""

OutputGrid = np.ndarray[(Any, Any, 3), np.float64]
"""Like normal grid, but next to IS_STREET and ALTITUDE, it contains also IS_RESIDUAL value."""

class TRAINING_GRID_INDICES(Enum):
    IS_PREDICTED = 0
    IS_STREET = GRID_INDICES.IS_STREET + 1
    ALTITUDE = GRID_INDICES.ALTITUDE + 1

class Model(ABC):

    def predict(self, input: GridManager[InputGrid]) -> GridManager[OutputGrid]:
        """Predicts grid for given input.

        Parameters
        ----------
        input : TraingingGrid
            Input grid. See TraingingGrid type description.

        Returns
        -------
        Grid
            Predicted output. Its size may be smaller than that of the input matrix.
        """
        raise NotImplementedError()
    
    def fit(self, input: GridManager[InputGrid], output: GridManager[OutputGrid]):
        """Fit model to the given data.

        Parameters
        ----------
        input : TraingingGrid
            Input grid. See TrainingGrid type description.
        output : Grid
            Expected output. Its size is equal the input size (except for the IS_PREDICTED dimension).
        """
        raise NotImplementedError()
    
    def _clean_input(self, input: InputGrid) -> InputGrid:
        """Cleans input grid from IS_STREET data, where IS_PREDICTED flag is on.

        Parameters
        ----------
        input : TraingingGrid
            Input grid.

        Returns
        -------
        TraingingGrid
            Shallow copy of given input grid with cleaned IS_STREET values, where IS_PREDICTED is on.
        """
        result = input.copy()
        rows_number, cols_number, _ = result.shape
        
        for row in range(rows_number):
            for col in range(cols_number):
                if result[row, col, TRAINING_GRID_INDICES.IS_PREDICTED] == 1:
                    result[row, col, TRAINING_GRID_INDICES.IS_STREET] = 0

        return result
    
    def _clean_output(self, input: InputGrid, output: OutputGrid) -> OutputGrid:
        """Cleans output grid from modifications, where IS_PREDICTED flag is off.

        Parameters
        ----------
        input : TraingingGrid
            Original input grid.

        Returns
        -------
        TraingingGrid
            Shallow copy of given output grid with copied data from input, where IS_PREDICTED is off.
        """
        result = output.copy()
        rows_number, cols_number, _ = result.shape
        
        for row in range(rows_number):
            for col in range(cols_number):
                if input[row, col, TRAINING_GRID_INDICES.IS_PREDICTED] == 0:
                    result[row, col, GRID_INDICES.IS_STREET] = input[row, col, TRAINING_GRID_INDICES.IS_STREET]
                    result[row, col, GRID_INDICES.ALTITUDE] = input[row, col, TRAINING_GRID_INDICES.ALTITUDE]

        return result
