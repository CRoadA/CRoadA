import pickle
import os.path
from typing import Any
from dataclasses import dataclass
import numpy as np
import random
import math
import tensorflow as tf

Sequence = tf.keras.utils.Sequence

from grid_manager import GridManager, GridType

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
        # Temporarily: one batch item -> cut size from the given list
        # OTHER POSSIBILITY: one batch -> one cut size
        self._cut_sequences_per_file = [
            CutSequence(file=self._files[i_file], cut_sizes=cut_sizes, sequence=self)
            for i_file in range(len(self._files))
        ]  # pre-create cut sequences for each file in the batch
        # I was thinking about making each iterator correspond to a specific part (segments) of each file
        # OR about reading firstly -> random parts of one file -> CutSequence must be quite deterministic because of the fact that Keras Sequence (like ClippingBatchSequence) relies on indexing to get the next batch - not remembering state

    def __len__(self) -> int:
        """Get number of batches per epoch."""
        return len(self._cut_sequences_per_file[0])

    def __getitem__(self, index: int) -> str:
        """Generate one batch of data == multiple files turned to cuts (a list of batch items, each being a list of cuts).
        Returns: batch - list of batch items (item == file), each being a list of cuts (start point and cut grid)."""
        batch = []
        # iter_len = len(self._iterators[index])
        iter_len = len(self._cut_sequences_per_file[0])
        for i in range(self._batch_size):
            # # index and i can be replaced by each other
            # batch.append(self._iterators[index].__getitem__(i % iter_len))
            batch.append(
                self._cut_sequences_per_file[i % len(self._files)].__getitem__(index % iter_len)
            )  # each iterator corresponds to one batch item

        return batch

    def batch_size(self) -> int:
        """Get the batch size."""
        return self._batch_size

    def cut_sizes(self) -> list[tuple[int, int]]:
        """Get the list of cut sizes."""
        return self._cut_sizes.copy()

    @staticmethod
    def cut_from_grid_segments(
        grid_manager: GridManager,
        cut_start_x: int,
        cut_start_y: int,
        cut_size: tuple[int, int],
        surplus: int = 0,
        clipping: bool = False,
    ) -> np.ndarray:
        """Create a cut grid from given grid manager by reading segments. No need to worry about cut_size exceeding boundaries - it is handled inside.

        Parameters
        ----------
        grid_manager : GridManager
            Grid manager to read segments from.
        cut_start_x : int
            Starting x coordinate of the cut.
        cut_start_y : int
            Starting y coordinate of the cut.
        cut_size : tuple[int, int]
            Size of the cut (rows, columns).
        surplus : int, optional
            Surplus size to consider around the cut, by default 0
        clipping : bool, optional
            Whether the cut is for clipping purposes, by default False
        Returns
        -------
        np.ndarray
            Cut grid as a numpy array.
        """
        # print(f"Requested cut at ({cut_start_x}, {cut_start_y}) of size {cut_size} with surplus {surplus}, clipping={clipping}")  # Debug print
        # Adjust surplus if it exceeds boundaries # TODO - Add padding if needed instead of reducing surplus
        surplus = max(min(surplus, cut_start_x, cut_start_y, cut_size[0], cut_size[1]), 0)

        # Load metadata
        metadata = grid_manager.get_metadata()

        if clipping:
            # Calculate start point with surplus
            cut_start_x = int(cut_start_x * (cut_size[0] - surplus) - (surplus / 2))
            cut_start_y = int(cut_start_y * (cut_size[1] - surplus) - (surplus / 2))
        else:
            cut_start_x = int(cut_start_x - surplus / 2)
            cut_start_y = int(cut_start_y - surplus / 2)
        cut_start_x = min(cut_start_x, metadata.columns_number - cut_size[0])
        cut_start_y = min(cut_start_y, metadata.rows_number - cut_size[1])

        # Calculate end point
        cut_end_x = min(cut_start_x + cut_size[0], metadata.columns_number)
        cut_end_y = min(cut_start_y + cut_size[1], metadata.rows_number)

        # Determine which segments to read
        segment_w, segment_h = metadata.segment_w, metadata.segment_h

        # print(f"Cut from segments: start_x={cut_start_x}, start_y={cut_start_y}, end_x={cut_end_x}, end_y={cut_end_y}")  # Debug print
        # print(f"Segment size: w={segment_w}, h={segment_h}")  # Debug print
        which_segment_start_x = int(cut_start_x // segment_w)
        which_segment_start_y = int(cut_start_y // segment_h)
        which_segment_end_x = int((cut_end_x - 1) // segment_w)
        which_segment_end_y = int((cut_end_y - 1) // segment_h)

        # print(f"Segments to read: start_x={which_segment_start_x}, start_y={which_segment_start_y}, end_x={which_segment_end_x}, end_y={which_segment_end_y}")  # Debug print
        cut_x = np.array([])  # Grid()
        cut_y = np.array([])
        # Go by segments and merge them into one bigger cut - first vertically, then horizontally.
        for indx_x in range(which_segment_start_x, which_segment_end_x + 1):
            for indx_y in range(which_segment_start_y, which_segment_end_y + 1):
                # TODO - memory issue -> adjust the maximal segment size or something => we will have proper segment sizes anyway
                segment_y = grid_manager.read_segment(indx_y, indx_x)
                # print(f"Reading segment at ({indx_x}, {indx_y}) with shape {segment_y.shape}")  # Debug print

                # Merge segment_y into cut_y vertically.
                if indx_y == which_segment_start_y and indx_y == which_segment_end_y:
                    cut_y = segment_y[cut_start_y % segment_h : cut_end_y % segment_h, :]
                elif indx_y == which_segment_start_y:
                    cut_y = segment_y[cut_start_y % segment_h :, :]
                elif indx_y == which_segment_end_y:
                    cut_y = np.vstack((cut_y, segment_y[: cut_end_y % segment_h, :]))
                else:
                    cut_y = np.vstack((cut_y, segment_y))

            # Merge cut_y into cut_x horizontally.
            if indx_x == which_segment_start_x and indx_x == which_segment_end_x:
                cut_x = cut_y[:, cut_start_x % segment_w : cut_end_x % segment_w]
            elif indx_x == which_segment_start_x:
                cut_x = cut_y[:, cut_start_x % segment_w :]
            elif indx_x == which_segment_end_x:
                cut_x = np.hstack((cut_x, cut_y[:, : cut_end_x % segment_w]))
            else:
                cut_x = np.hstack((cut_x, cut_y))

        return cut_x

    @staticmethod
    def write_cut_to_grid_segments(
        cut: np.ndarray,
        cut_size: tuple[int, int],
        segment_w: int,
        segment_h: int,
        cut_start_x: int,
        cut_start_y: int,
        from_file_path: str,
        to_directory: str,
        grid_type: GridType,
    ) -> GridManager:
        """
        Write the cut grid into a new GridManager by segments.

        :param cut: The cut grid to write.
        :type cut: np.ndarray
        :param cut_size: Size of the cut (rows, columns).
        :type cut_size: tuple[int, int]
        :param segment_w: Segment width.
        :type segment_w: int
        :param segment_h: Segment height.
        :type segment_h: int
        :param cut_start_x: Starting x-coordinate of the cut.
        :type cut_start_x: int
        :param cut_start_y: Starting y-coordinate of the cut.
        :type cut_start_y: int
        :param from_file_path: File path where the cut grid comes from.
        :type from_file_path: str
        :param grid_type: Type of the grid.
        :type grid_type: GridType
        :return: A new GridManager instance containing the cut grid.
        :rtype: GridManager
        """
        cut_segment_rows = math.ceil(cut_size[0] / segment_h)  # number of segments in cut vertically
        cut_segment_columns = math.ceil(cut_size[1] / segment_w)  # number of segments in cut horizontally

        # Check if the cut grid file already exists
        file_path = os.path.join(
            to_directory, f"{from_file_path}_cut_{cut_start_x}_{cut_start_y}_{cut_size[0]}_{cut_size[1]}.dat"
        )
        cut_grid: GridManager = None

        if os.path.isfile(file_path):
            cut_grid = GridManager(file_path)  # load grid manager
        else:
            cut_grid = GridManager[grid_type](
                f"{from_file_path}_cut_{cut_start_x}_{cut_start_y}_{cut_size[0]}_{cut_size[1]}.dat",
                cut_size[0],
                cut_size[1],
                0,
                0,
                1,
                segment_h,
                segment_w,
                to_directory,
            )
            for segment_row in range(cut_segment_rows):
                for segment_col in range(cut_segment_columns):
                    cut_grid.write_segment(
                        cut[
                            segment_row * segment_h : min((segment_row + 1) * segment_h, cut_size[0]),
                            segment_col * segment_w : min((segment_col + 1) * segment_w, cut_size[1]),
                        ],
                        segment_row,
                        segment_col,
                    )
        return cut_grid


@dataclass
class CutSequence(Sequence):
    """Sequence that yields random cuts from a file in a BatchSequence."""

    def __init__(self, file, cut_sizes: list[tuple[int, int]], sequence: BatchSequence):
        """
        Initialize cut sequence from a file - cuts will be parts of the file with sizes from cut_sizes.

        :param file: File path to create cuts from.
        :param cut_sizes: List of possible cut sizes (rows, columns) to use when generating cuts from the file.
        :type cut_sizes: list[tuple[int, int]]
        :param sequence: BatchSequence instance this CutSequence belongs to.
        :type sequence: BatchSequence
        """
        self._file_path = file
        # sequence._files[sequence._files.index(self._file_path)] += 1  # increment count of uses # TODO - not important
        self._cut_sizes = cut_sizes

        self._grid_manager = GridManager(self._file_path)  # load grid manager
        self._grid_metadata = self._grid_manager.get_metadata()
        self._grid_rows, self._grid_cols = self._grid_metadata.rows_number, self._grid_metadata.columns_number
        self._segment_rows, self._segment_cols = self._grid_metadata.segment_h, self._grid_metadata.segment_w

        self._max_x = self._grid_rows - min(cut[0] for cut in self._cut_sizes)  # max starting x for cut
        self._max_y = self._grid_cols - min(cut[1] for cut in self._cut_sizes)  # max starting y for cut
        self._already_used = list()  # list of already used starting points for cuts with their sizes

    def __len__(self) -> int:
        """Get number of iterations possible"""
        return (
            self._max_x * self._max_y
        )  # not necessary, because we have it random; possible duplicates due to time complexity

    def __getitem__(self, index: int) -> tuple[tuple[int, int], GridManager]:
        """Get the next random cut (part of a batch item) from the file (a batch item == multiple cuts)."""
        # Choose cut size - not random because of determinism
        cut_size = self._cut_sizes[index % len(self._cut_sizes)]
        if cut_size[0] > self._grid_rows or cut_size[1] > self._grid_cols:
            cut_size = (self._grid_rows, self._grid_cols)

        self._max_x = self._grid_cols - cut_size[0]  # max starting x for cut
        self._max_y = self._grid_rows - cut_size[1]  # max starting y for cut
        print(f"Cut size: {cut_size}, Max start points: x={self._max_x}, y={self._max_y}")  # Debug print

        # Choose random starting point
        cut_start_x = random.randint(0, self._max_x)
        cut_start_y = random.randint(0, self._max_y)

        self._already_used.append(((cut_start_x, cut_start_y), cut_size))

        # Create cut grid
        cut = BatchSequence.cut_from_grid_segments(
            self._grid_manager, cut_start_x, cut_start_y, cut_size, surplus=0, clipping=False
        )

        cut = np.copy(cut)
        cut = np.resize(
            cut, (cut_size[0], cut_size[1], cut.shape[2] + 1)
        )  # is modifiable # TODO - check IS_PREDICTED case

        cut_grid = BatchSequence.write_cut_to_grid_segments(
            cut,
            cut_size,
            self._grid_metadata.segment_w,
            self._grid_metadata.segment_h,
            cut_start_x,
            cut_start_y,
            self._file_path,
            "./tmp/batches/batch_sequences/cuts/",
            InputGrid,
        )

        return ((cut_start_x, cut_start_y), cut_grid)

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
