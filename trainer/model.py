from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Any
import numpy as np
import os

Sequence = tf.keras.utils.Sequence

from grid_manager import GRID_INDICES, GridManager

InputGrid = np.ndarray[(Any, Any, 4), np.float64]
PredictGrid = np.ndarray[(Any, Any, 3), np.float64]


class TRAINING_GRID_INDICES:
    IS_PREDICTED = 0
    IS_STREET = GRID_INDICES.IS_STREET + 1
    ALTITUDE = GRID_INDICES.ALTITUDE + 1
    IS_RESIDENTIAL = GRID_INDICES.IS_RESIDENTIAL + 1


class PREDICT_GRID_INDICES(GRID_INDICES):
    pass


class Model(ABC):
    def __init__(self, dir: str | None = None):
        if dir is not None:
            self._dir = dir
        else:
            self._dir = "./models/created_at_" + str(tf.timestamp())

        os.makedirs(self._dir, exist_ok=True)

    def predict(self, input: GridManager[InputGrid]) -> GridManager[PredictGrid]:
        """Predicts grid for given input.

        Parameters
        ----------
        input : TrainingGrid
            Input grid. See TrainingGrid type description.

        Returns
        -------
        Grid
            Predicted output. Its size may be smaller than that of the input matrix.
        """
        raise NotImplementedError()

    def fit(self):
        """Fit model to the given data."""
        raise NotImplementedError()

    @abstractmethod
    def get_input_grid_surplus(self) -> int:
        """Get the surplus size of input grid compared to output grid.

        Returns
        -------
        int
            Number of rows/columns that the output grid is smaller than the input grid.
        """
        raise NotImplementedError()

    @staticmethod
    def clean_input(input: InputGrid, input_third_dimension: int) -> InputGrid:
        """Cleans input grid from IS_STREET data, where IS_PREDICTED flag is on.

        Parameters
        ----------
        input : TrainingGrid
            Input grid.

        Returns
        -------
        TrainingGrid
            Shallow copy of given input grid with cleaned IS_STREET values, where IS_PREDICTED is on.
        """
        result = input.copy()
        rows_number, cols_number, _ = result.shape

        for row in range(rows_number):
            for col in range(cols_number):
                if result[row, col, TRAINING_GRID_INDICES.IS_PREDICTED] == 1:
                    result[row, col, TRAINING_GRID_INDICES.IS_STREET] = 0
                    if input_third_dimension >= 4:
                        result[row, col, TRAINING_GRID_INDICES.IS_RESIDENTIAL] = 0

        return result

    def _clean_output(self, input: InputGrid, output: PredictGrid, output_third_dimension: int) -> PredictGrid:
        """Cleans output grid from modifications, where IS_PREDICTED flag is off.

        Parameters
        ----------
        input : TrainingGrid
            Original input grid.

        Returns
        -------
        TrainingGrid
            Shallow copy of given output grid with copied data from input, where IS_PREDICTED is off.
        """
        result = output.copy()
        rows_number, cols_number, _ = result.shape

        for row in range(rows_number):
            for col in range(cols_number):
                if input[row, col, TRAINING_GRID_INDICES.IS_PREDICTED] == 0:
                    result[row, col, GRID_INDICES.IS_STREET] = input[row, col, TRAINING_GRID_INDICES.IS_STREET]
                    if output_third_dimension >= 2:
                        result[row, col, GRID_INDICES.ALTITUDE] = input[row, col, TRAINING_GRID_INDICES.ALTITUDE]
                    if output_third_dimension >= 3:
                        result[row, col, GRID_INDICES.IS_RESIDENTIAL] = input[
                            row, col, TRAINING_GRID_INDICES.IS_RESIDENTIAL
                        ]

        return result

    @abstractmethod
    def save(self):
        """Saves the model to the specified path.

        Parameters
        ----------
        path : str
            Path to save the model.
        """
        raise NotImplementedError()
