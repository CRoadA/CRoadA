from abc import ABC, abstractmethod
from enum import Enum
import tensorflow as tf
from typing import Any
import numpy as np

Sequence = tf.keras.utils.Sequence

from grid_manager import GRID_INDICES, GridManager
import trainer.batch_sequence as batch_sequence

InputGrid = np.ndarray[(Any, Any, 3), np.float64]
OutputGrid = np.ndarray[(Any, Any, 3), np.float64]

class TRAINING_GRID_INDICES:
    IS_PREDICTED = 0
    IS_STREET = GRID_INDICES.IS_STREET + 1
    ALTITUDE = GRID_INDICES.ALTITUDE + 1


class Model(ABC):
    def __init__(self, dir: str | None = None):
        if dir is not None:
            self._dir = dir
        else:
            self._dir = "./models/created_at_" + str(tf.timestamp())
        

    def predict(self, input: GridManager[batch_sequence.InputGrid]) -> GridManager[batch_sequence.OutputGrid]:
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

    def fit(self, input: batch_sequence.BatchSequence, epochs: int):
        """Fit model to the given data.

        Parameters
        ----------
        input : batch_sequence.BatchSequence
            Input batch sequence. See BatchSequence type description.
        #output : Grid
            Expected output. Its size is equal the input size (except for the IS_PREDICTED dimension).
        """
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
    def clean_input(input: batch_sequence.InputGrid) -> batch_sequence.InputGrid:
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

        return result

    def _clean_output(
        self, input: batch_sequence.InputGrid, output: batch_sequence.OutputGrid
    ) -> batch_sequence.OutputGrid:
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
                    result[row, col, GRID_INDICES.ALTITUDE] = input[row, col, TRAINING_GRID_INDICES.ALTITUDE]

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
