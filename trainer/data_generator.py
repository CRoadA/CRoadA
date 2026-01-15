import math
import random
from typing import Any
import numpy as np

from grid_manager import GridManager
from trainer.cut_grid import cut_from_grid_segments, write_cut_to_grid_segments
from trainer.model import Model

InputGrid = np.ndarray[(Any, Any, 3), np.float64]
"""Like normal Grid, but with bools indicating, if it should be changed (the 0-th coordinate of the thrid dimension). If it is False, then the IS_STREET bool is treated as zero."""

OutputGrid = np.ndarray[(Any, Any, 3), np.float64]
"""Like normal grid, but next to IS_STREET and ALTITUDE, it contains also IS_RESIDENTIAL value."""



def clipping_sample_generator(files: list[str], cut_sizes: list[tuple[int, int]], clipping_size: int, input_surplus: int):
    """Generates samples for clipping training.
    Yields tuples (input, output), where:
    - input is a numpy array of shape (clipping_size, clipping_size, 2) containing IS_STREET and ALTITUDE values,
    - output is a dictionary with keys 'is_street' and 'altitude', each being a numpy array of shape (clipping_size - input_surplus, clipping_size - input_surplus, 1).
    """
    while True:
        file = random.choice(files)
        cut_size = random.choice(cut_sizes)

        _, cut_grid = generate_cut(file, cut_size)
        metadata = cut_grid.get_metadata()

        max_x = metadata.columns_number - clipping_size
        max_y = metadata.rows_number - clipping_size
        clipping_x = random.randint(0, max_x)
        clipping_y = random.randint(0, max_y)

        clipping = cut_from_grid_segments(
            cut_grid,
            clipping_x,
            clipping_y,
            (clipping_size, clipping_size),
            input_surplus,
            clipping=True,
        )

        # TODO - when to clean? - (w uczeniu czasem powinien dostawać nie w pełni wyczyszczone dane czy nie?)
        clipping = Model.clean_input(clipping)
        x = clipping[:, :, 0:2] # TODO - without IS_RESIDENTIAL
        output_clipping = clipping[
            int(input_surplus / 2) : clipping_size - int(input_surplus / 2),
            int(input_surplus / 2) : clipping_size - int(input_surplus / 2),
            :,
        ]
        # TODO - adjust IS_REGIONAL value - probably we meant local, residential roads - IS_RESIDENTIAL value
        y_is_street = output_clipping[:, :, 0:1] # shape: (cut_size - input_surplus, cut_size - input_surplus, 1)
        y_altitude = output_clipping[:, :, 1:2] # shape: (cut_size - input_surplus, cut_size - input_surplus, 1)
        yield x, {"is_street": y_is_street, "altitude": y_altitude}


def generate_clippingbatch(files: list[str], batch_size: int, cut_sizes: list[tuple[int, int]], clipping_size: int, input_surplus: int) -> tuple[np.ndarray, np.ndarray]:
    batch_x = []
    batch_y_is_street = []
    batch_y_altitude = []

    for _ in range(batch_size):
        file = random.choice(files)
        cut_size = random.choice(cut_sizes)

        _, cut_grid = generate_cut(file, cut_size)
        metadata = cut_grid.get_metadata()

        max_x = metadata.columns_number - clipping_size
        max_y = metadata.rows_number - clipping_size
        clipping_x = random.randint(0, max_x)
        clipping_y = random.randint(0, max_y)

        clipping = cut_from_grid_segments(
            cut_grid,
            clipping_x,
            clipping_y,
            (clipping_size, clipping_size),
            input_surplus,
            clipping=True,
        )

        # TODO - when to clean? - (w uczeniu czasem powinien dostawać nie w pełni wyczyszczone dane czy nie?)
        clipping = Model.clean_input(clipping)
        batch_x.append(clipping[:, :, 0:2]) # TODO - without IS_RESIDENTIAL
        output_clipping = clipping[
            int(input_surplus / 2) : clipping_size - int(input_surplus / 2),
            int(input_surplus / 2) : clipping_size - int(input_surplus / 2),
            :,
        ]
        # TODO - adjust IS_REGIONAL value - probably we meant local, residential roads - IS_RESIDENTIAL value
        batch_y_is_street.append(output_clipping[:, :, 0:1]) # shape: (cut_size - input_surplus, cut_size - input_surplus, 1)
        batch_y_altitude.append(output_clipping[:, :, 1:2]) # shape: (cut_size - input_surplus, cut_size - input_surplus, 1)

    batch_x = np.stack(batch_x)
    batch_y_is_street = np.stack(batch_y_is_street)
    batch_y_altitude = np.stack(batch_y_altitude)

    return batch_x, {"is_street": batch_y_is_street, "altitude": batch_y_altitude}


def generate_cutbatch(files: list[str], index: int, batch_size: int, cut_sizes: list[tuple[int, int]]):
    """Generate one batch of data == multiple files turned to cuts (a list of batch items, each being a list of cuts).
    Returns: batch - list of batch items (item == file), each being a list of cuts (start point and cut grid)."""
    
    batch = []
    for i_file in range(batch_size):
        batch.append(generate_cut(index, files[i_file], batch_size, cut_sizes))

    return batch

def generate_cut(file: str, cut_size: tuple[int, int]) -> tuple[tuple[int, int], GridManager]:
    """Get the next random cut (part of a batch item) from the file (a batch item == multiple cuts)."""
    grid_manager = GridManager(file)  # load grid manager
    grid_metadata = grid_manager.get_metadata()
    grid_rows, grid_cols = grid_metadata.rows_number, grid_metadata.columns_number

    max_x = grid_rows - cut_size[0]  # max starting x for cut
    max_y = grid_cols - cut_size[1]  # max starting y for cut

    if cut_size[0] > grid_rows or cut_size[1] > grid_cols:
        cut_size = (grid_rows, grid_cols)

    max_x = grid_cols - cut_size[0]  # max starting x for cut
    max_y = grid_rows - cut_size[1]  # max starting y for cut

    # Choose random starting point
    cut_start_x = random.randint(0, max_x)
    cut_start_y = random.randint(0, max_y)

    # Create cut grid
    cut = cut_from_grid_segments(grid_manager, cut_start_x, cut_start_y, cut_size, surplus=0, clipping=False)

    cut = np.copy(cut)
    cut = np.resize(cut, (cut_size[0], cut_size[1], cut.shape[2] + 1))  # is modifiable # TODO - check IS_PREDICTED case

    cut_grid = write_cut_to_grid_segments(
        cut,
        cut_size,
        grid_metadata.segment_w,
        grid_metadata.segment_h,
        cut_start_x,
        cut_start_y,
        file,
        "./tmp/batches/batch_sequences/cuts/",
        InputGrid
    )

    return ((cut_start_x, cut_start_y), cut_grid)