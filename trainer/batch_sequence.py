from typing import Any, Iterator
from dataclasses import dataclass
import numpy as np
import random
import math
import tensorflow as tf

Sequence = tf.keras.utils.Sequence

from CRoadA.grid_manager import Grid, GridManager

InputGrid = np.ndarray[(Any, Any, 3), np.float64]
"""Like normal Grid, but with bools indicating, if it should be changed (the 0-th coordinate of the thrid dimension). If it is False, then the IS_STREET bool is treated as zero."""

OutputGrid = np.ndarray[(Any, Any, 3), np.float64]
"""Like normal grid, but next to IS_STREET and ALTITUDE, it contains also IS_RESIDENTIAL value."""


@dataclass
class BatchSequence(Sequence):
    """Keras Sequence that yields a batch of cuts from possibly different files. It is later wrapped by ClippingBatchSequence to provide clipped grids for training.
    Using Sequence improves performance by enabling the possibility of multi-threading and pre-fetching of data."""

    def __init__(self, files: list[str], batch_size: int, cut_sizes: list[tuple[int, int]]):
        """
        Create a batch sequence from given files.
        
        :param files: Files to prepare batches from.
        :type files: list[str]
        :param batch_size: Number of items in each batch - each item corresponds to one file.
        :type batch_size: int
        :param cut_sizes: List of possible cut sizes (rows, columns) to use when generating cuts from files.
        :type cut_sizes: list[tuple[int, int]]
        """
        self._files = files
        self._cut_sizes = cut_sizes
        self._batch_size = batch_size
        # Temporarily: one batch item -> random cut size from the given list
        # OTHER POSSIBILITY: one batch -> one cut size
        self._iterators = [
            CutSequence(file=self._files[i_file], cut_sizes=cut_sizes, sequence=self) for i_file in range(batch_size)
        ]  # pre-create iterators
        # I was thinking about making each iterator correspond to a specific part (segments) of each file
        # OR about reading firstly -> random parts of one file

    def __len__(self):
        """Get number of batches per epoch."""
        return len(self._iterators[0])

    def __getitem__(self, index: int) -> list[list[tuple[tuple[int, int], GridManager]]]:
        """Generate one batch of data == multiple files turned to cuts (a list of batch items, each being a list of cuts).
        Returns: batch - list of batch items (item == file), each being a list of cuts (start point and cut grid)."""
        batch = []
        # iter_len = len(self._iterators[index])
        iter_len = len(self._iterators[0])
        for i in range(self._batch_size):
            # # index and i can be replaced by each other
            # batch.append(self._iterators[index].__getitem__(i % iter_len))
            batch.append(
                self._iterators[i].__getitem__(index % iter_len)
            )  # each iterator corresponds to one batch item

        return batch

    def batch_size(self) -> int:
        """Get the batch size."""
        return self._batch_size

    def cut_sizes(self) -> list[tuple[int, int]]:
        """Get the list of cut sizes."""
        return self._cut_sizes


@dataclass
class CutSequence(Sequence):
    """Sequence that yields random cuts from a random file in a BatchSequence."""

    def __init__(self, file, cut_sizes: list[tuple[int, int]], sequence: BatchSequence):
        """
        Initialize cut sequence from a file - cuts will be parts of the file with sizes from cut_sizes.
        
        :param file: File path to create cuts from.
        :param cut_sizes: List of possible cut sizes (rows, columns) to use when generating cuts from the file.
        :type cut_sizes: list[tuple[int, int]]
        :param sequence: BatchSequence instance this CutSequence belongs to.
        :type sequence: BatchSequence
        """
        #self._file_path = random.choice(list(sequence._files.keys())) # TODO - idk if this works as intended for fit not remembering the sequence - maybe should be more deterministic - MOVE RANDOMNESS OUTSIDE
        self._file_path = file
        sequence._files[self._file_path] += 1  # increment count of uses
        self._cut_sizes = cut_sizes

        self._grid_manager = GridManager(self._file_path)  # load grid manager
        self._grid_metadata = self._grid_manager.get_metadata()
        self._grid_rows, self._grid_cols = self._grid_metadata.rows_number, self._grid_metadata.columns_number
        self._segment_rows, self._segment_cols = self._grid_metadata.segment_h, self._grid_metadata.segment_w

        self._max_x = self._grid_rows - min(cut[0] for cut in self._cut_sizes)  # max starting x for cut
        self._max_y = self._grid_cols - min(cut[1] for cut in self._cut_sizes)  # max starting y for cut
        self._already_used = list()  # list of already used starting points for cuts with their sizes

    def __len__(self):
        """Get number of iterations possible"""
        return (
            self._max_x * self._max_y
        )  # not necessary, because we have it random; possible duplicates due to time complexity

    def __getitem__(self, index: int) -> tuple[tuple[int, int], GridManager]:
        """Get the next random cut (part of a batch item) from the file (a batch item == multiple cuts)."""
        # Choose random cut size
        cut_size = self._cut_sizes[index % len(self._cut_sizes)]
        if cut_size[0] > self._grid_rows or cut_size[1] > self._grid_cols:
            cut_size = (self._grid_rows, self._grid_cols)

        self._max_x = self._grid_rows - cut_size[0]  # max starting x for cut
        self._max_y = self._grid_cols - cut_size[1]  # max starting y for cut

        # Choose random starting point
        start_x = random.randint(0, self._max_x)
        start_y = random.randint(0, self._max_y)

        self._already_used.append(((start_x, start_y), cut_size))

        # Calculate end point
        end_x = start_x + cut_size[0]
        end_y = start_y + cut_size[1]

        # Determine which segments to read
        which_segment_start_x = start_x // self._segment_rows
        which_segment_start_y = start_y // self._segment_cols
        which_segment_end_x = end_x // self._segment_rows
        which_segment_end_y = end_y // self._segment_cols

        cut_x = np.array([])  # Grid()
        # Go by segments and merge them into one bigger cut - first vertically, then horizontally.
        for indx_x in range(which_segment_start_x, which_segment_end_x + 1):
            for indx_y in range(which_segment_start_y, which_segment_end_y + 1):
                # TODO - memory issue
                segment_y = self._grid_manager.read_segment(indx_x, indx_y)
                # Merge segment_y into cut_y vertically.
                if indx_y == which_segment_start_y:
                    cut_y = segment_y
                else:
                    cut_y = np.vstack((cut_y, segment_y))

            # Merge cut_y into cut_x horizontally.
            if indx_x == which_segment_start_x:
                cut_x = cut_y
            else:
                cut_x = np.hstack((cut_x, cut_y))

        cut_x = cut_x.resize(1, (cut_size[0], cut_size[1], cut_x.shape[2] + 1))  # is modifiable
        cut_segment_rows = math.ceil(
            cut_size[0] / self._grid_metadata.segment_h
        )  # number of segments in cut vertically
        cut_segment_columns = math.ceil(
            cut_size[1] / self._grid_metadata.segment_w
        )  # number of segments in cut horizontally
        cut_grid = GridManager[InputGrid](
            f"{self._file_path}_cut_{start_x}_{start_y}_{cut_size[0]}_{cut_size[1]}.dat",
            cut_size[0],
            cut_size[1],
            0,
            0,
            None,
            self._grid_metadata.segment_h,
            self._grid_metadata.segment_w,
            "./tmp/batches/batch_sequences/cuts/",
        )
        for segment_row in range(cut_segment_rows):
            for segment_col in range(cut_segment_columns):
                cut_grid.write_segment(
                    cut_x[
                        segment_row
                        * self._grid_metadata.segment_h : min(
                            (segment_row + 1) * self._grid_metadata.segment_h, cut_size[0]
                        ),
                        segment_col
                        * self._grid_metadata.segment_w : min(
                            (segment_col + 1) * self._grid_metadata.segment_w, cut_size[1]
                        ),
                    ],
                    segment_row,
                    segment_col,
                )

        return ((start_x, start_y), cut_grid)

    def count_used_start_points(self) -> int:
        """Count the number of used starting points for cuts (with duplicates)."""
        return len(self._already_used)

    def count_unique_start_points(self) -> int:
        """Count the number of unique used starting points for cuts (without duplicates)."""
        return len(set(self._already_used[:, 0]))

    def calculate_area_covered(self) -> float:
        """Calculate the area covered by the cuts made so far as a fraction of the total file area."""
        total_area = self._grid_rows * self._grid_cols
        all_points = [0 for _ in range(self._grid_rows)] * [0 for _ in range(self._grid_cols)]
        for (start_x, start_y), cut_size in set(self._already_used):
            for i in range(cut_size[0]):
                for j in range(cut_size[1]):
                    all_points[start_x + i][start_y + j] = 1
        covered_area = sum([sum(row) for row in all_points])
        return covered_area / total_area
