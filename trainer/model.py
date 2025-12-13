from abc import ABC
from enum import Enum
import tensorflow as tf
import math
import numpy as np

Sequence = tf.keras.utils.Sequence

from grid_manager import GRID_INDICES, Grid, GridManager
import data_types


class TRAINING_GRID_INDICES(Enum):
    IS_PREDICTED = 0
    IS_STREET = GRID_INDICES.IS_STREET + 1
    ALTITUDE = GRID_INDICES.ALTITUDE + 1


class Model(ABC):

    def predict(self, input: GridManager[data_types.InputGrid]) -> GridManager[data_types.OutputGrid]:
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

    def fit(self, input: data_types.BatchSequence, output: GridManager[data_types.OutputGrid]):
        """Fit model to the given data.

        Parameters
        ----------
        input : TraingingGrid
            Input grid. See TrainingGrid type description.
        output : Grid
            Expected output. Its size is equal the input size (except for the IS_PREDICTED dimension).
        """
        raise NotImplementedError()

    def get_input_grid_surplus(self) -> int:
        """Get the surplus size of input grid compared to output grid.

        Returns
        -------
        int
            Number of rows/columns that the output grid is smaller than the input grid.
        """
        raise NotImplementedError()

    @staticmethod
    def clean_input(input: data_types.InputGrid) -> data_types.InputGrid:
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

    def _clean_output(self, input: data_types.InputGrid, output: data_types.OutputGrid) -> data_types.OutputGrid:
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


class ClippingBatchSequence(Sequence):
    """Batch sequence that clips input and output grids to proper sizes for the model."""

    def __init__(self, base_sequence: data_types.BatchSequence, clipping_size: int, input_grid_surplus: int):
        """Initializes clipping batch sequence.

        Parameters
        ----------
        base_sequence : BatchSequence
            Base batch sequence to clip data from.
        clipping_size : tuple[int, int]
            Size to clip the input and output grids to.
        input_grid_surplus : int
            Surplus size of input grid compared to output grid.
        """
        self._clipping_size = clipping_size
        self._input_surplus = input_grid_surplus
        self._batch_index = 0
        self._clipping_index = 0
        self._batches = list(base_sequence)

    def __len__(self) -> int:
        result = 0
        for batch in self._batches:
            start_point, cut = batch[0]
            metadata = cut.get_metadata()
            rows_number, cols_number = (
                metadata.rows_number - self._input_surplus,
                metadata.columns_number - self._input_surplus,
            )
            result += math.ceil(rows_number / (self._clipping_size - self._input_surplus)) * math.ceil(
                cols_number / (self._clipping_size - self._input_surplus)
            )

        return result

    def __getitem__(self, index: int) -> tuple[data_types.InputGrid, data_types.OutputGrid]:
        batch_x = []
        batch_y = []

        _, cut_grid = self._batches[self._batch_index][0]
        metadata = cut_grid.get_metadata()
        clipping_rows, clipping_cols = (
            math.ceil((metadata.rows_number - self._input_surplus) / (self._clipping_size - self._input_surplus)),
            math.ceil((metadata.columns_number - self._input_surplus) / (self._clipping_size - self._input_surplus)),
        )

        for _, cut_grid in self._batches[self._batch_index]:
            metadata = cut_grid.get_metadata()
            clipping_x = self._clipping_index % clipping_cols
            clipping_y = self._clipping_index // clipping_cols

            clipping_start_x = clipping_x * (self._clipping_size - self._input_surplus) - (self._input_surplus / 2)
            clipping_start_y = clipping_y * (self._clipping_size - self._input_surplus) - (self._input_surplus / 2)

            segment_h, segment_w = metadata.segment_h, metadata.segment_w

            which_segment_start_x = clipping_start_x // segment_h
            which_segment_start_y = clipping_start_y // segment_w

            which_segment_end_x = (clipping_start_x + self._clipping_size) // segment_h
            which_segment_end_y = (clipping_start_y + self._clipping_size) // segment_w

            clipping_x = np.array([])  # Grid()
            # Go by segments and merge them into one bigger cut - first vertically, then horizontally.
            for indx_x in range(which_segment_start_x, which_segment_end_x + 1):
                for indx_y in range(which_segment_start_y, which_segment_end_y + 1):
                    segment_y = cut_grid.read_segment(indx_x, indx_y)
                    # Merge segment_y into cut_y vertically.
                    if indx_y == which_segment_start_y:
                        clipping_y = segment_y
                    else:
                        clipping_y = np.vstack((clipping_y, segment_y))

                # Merge clipping_y into clipping_x horizontally.
                if indx_x == which_segment_start_x:
                    clipping_x = clipping_y
                else:
                    clipping_x = np.hstack((clipping_x, clipping_y))

            clipping_x = Model.clean_input(clipping_x)
            batch_x.append(clipping_x)
            output_clipping_x = clipping_x[
                int(self._input_surplus / 2) : self._clipping_size - int(self._input_surplus / 2),
                int(self._input_surplus / 2) : self._clipping_size - int(self._input_surplus / 2),
                :,
            ]
            # TODO - adjust IS_REGIONAL value
            batch_y.append(output_clipping_x[:, :, 0:2])  # only IS_STREET and ALTITUDE

        self._clipping_index += 1
        if clipping_rows * clipping_cols == self._clipping_index:
            self._clipping_index = 0
            self._batch_index += 1
            if self._batch_index >= len(self._batches):
                self._batch_index = 0

        return batch_x, batch_y
