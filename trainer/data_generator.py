import random
from typing import Any
import numpy as np
import tensorflow as tf

from grid_manager import GridManager
from trainer.cut_grid import cut_from_grid_segments, write_cut_to_grid_segments
from trainer.model import Model

InputGrid = np.ndarray[(Any, Any, 3), np.float32]
"""Like normal Grid, but with bools indicating, if it should be changed (the 0-th coordinate of the thrid dimension). If it is False, then the IS_STREET bool is treated as zero."""

OutputGrid = np.ndarray[(Any, Any, 3), np.float32]
"""Like normal grid, but next to IS_STREET and ALTITUDE, it contains also IS_RESIDENTIAL value."""

def get_tf_dataset(files: list[str], cut_sizes: list[tuple[int, int]], clipping_size: int, input_surplus: int, batch_size: int, third_dimension: int = 3) -> tf.data.Dataset:
    """Get TensorFlow dataset from clipping sample generator.
    Each sample is a tuple (input, output), where:
    - input is a numpy array of shape (clipping_size, clipping_size, 3) containing IS_STREET, ALTITUDE and IS_MODIFIABLE values,
    - output is a dictionary with keys 'is_street' and 'altitude', each being a numpy array of shape (clipping_size - input_surplus, clipping_size - input_surplus, 1).
    """
    # Define output signature for the dataset - helps TensorFlow understand the shape and type of the data
    output_signature = (
        tf.TensorSpec(shape=(clipping_size, clipping_size, third_dimension), dtype=tf.float32),
        {
            "is_street": tf.TensorSpec(shape=(clipping_size - input_surplus, clipping_size - input_surplus, 1), dtype=tf.float32),
            "altitude": tf.TensorSpec(shape=(clipping_size - input_surplus, clipping_size - input_surplus, 1), dtype=tf.float32),
            # TODO - add IS_RESIDENTIAL when data is ready
        },
    )

    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        lambda: clipping_sample_generator(files, cut_sizes, clipping_size, input_surplus),
        output_signature=output_signature,
    )

    # Batch and prefetch the dataset - to improve performance  
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

def clipping_sample_generator(files: list[str], cut_sizes: list[tuple[int, int]], clipping_size: int, input_surplus: int):
    """Generates samples for clipping training.
    Yields tuples (input, output), where:
    - input is a numpy array of shape (clipping_size, clipping_size, 3) containing IS_STREET, ALTITUDE and IS_MODIFIABLE values,
    - output is a dictionary with keys 'is_street' and 'altitude', each being a numpy array of shape (clipping_size - input_surplus, clipping_size - input_surplus, 1).
    """
    while True:
        # Select a random file and cut size
        file = random.choice(files)
        cut_size = random.choice(cut_sizes)

        # Get a random cut from the file
        _, cut_grid = generate_cut(file, cut_size)
        metadata = cut_grid.get_metadata()

        # Determine random clipping position within the cut grid
        max_x = metadata.columns_number - clipping_size
        max_y = metadata.rows_number - clipping_size
        clipping_x = random.randint(0, max_x)
        clipping_y = random.randint(0, max_y)

        # Create the clipping from the cut grid
        clipping = cut_from_grid_segments(
            cut_grid,
            clipping_x,
            clipping_y,
            (clipping_size, clipping_size),
            input_surplus,
            clipping=True,
        )

        # Prepare input and output for the model

        # Clean the clipping from IS_STREET data where IS_PREDICTED flag is on
        clipping = Model.clean_input(clipping)
        # without IS_RESIDENTIAL, but with IS_MODIFIABLE
        x = clipping[:, :, 0:3].astype(np.float32) # Keras does not like float64
        # Fill IS_MODIFIABLE channel with ones -> we want to use all data for training
        x[:, :, 2] = 1

        # Prepare the area which we expect the model to predict
        output_clipping = clipping[
            int(input_surplus / 2) : clipping_size - int(input_surplus / 2),
            int(input_surplus / 2) : clipping_size - int(input_surplus / 2),
            :,
        ]

        # Prepare output values
        y_is_street = output_clipping[:, :, 0:1].astype(np.float32) # shape: (cut_size - input_surplus, cut_size - input_surplus, 1)
        y_altitude = output_clipping[:, :, 1:2].astype(np.float32) # shape: (cut_size - input_surplus, cut_size - input_surplus, 1)
        # TODO - add y_is_residential when data is ready

        # Yield the input-output pair - TensorFlow will handle batching
        yield x, {"is_street": y_is_street, "altitude": y_altitude}

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
    # Ensure the cut has the correct number of channels (including IS_MODIFIABLE channel)
    cut = np.resize(cut, (cut_size[0], cut_size[1], cut.shape[2] + 1))

    # Write cut to a temporary GridManager to return
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

    # Return the cut starting position and the cut grid manager
    return ((cut_start_x, cut_start_y), cut_grid)