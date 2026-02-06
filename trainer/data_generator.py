import random
from typing import Any
import numpy as np
import tensorflow as tf

from grid_manager import Grid, GridManager, GRID_INDICES
from trainer.cut_grid import cut_from_cut, cut_from_grid_segments
from trainer.model import TRAINING_GRID_INDICES, PREDICT_GRID_INDICES, Model

InputGrid = np.ndarray[(Any, Any, 3), np.float32]
"""Like normal Grid, but with bools indicating, if it should be changed (the 0-th coordinate of the thrid dimension). If it is False, then the IS_STREET bool is treated as zero."""

OutputGrid = np.ndarray[(Any, Any, 3), np.float32]
"""Like normal grid, but next to IS_STREET and ALTITUDE, it contains also IS_RESIDENTIAL value."""


def get_tf_dataset(
    files: list[GridManager[Grid]],
    cut_sizes: list[tuple[int, int]],
    clipping_size: int,
    input_surplus: int,
    batch_size: int,
    input_third_dimension: int = 3,
    output_third_dimension: int = 3,
) -> tf.data.Dataset:
    """Get TensorFlow dataset from clipping sample generator.
    Each sample is a tuple (input, output), where:
    - input is a numpy array of shape (clipping_size, clipping_size, 3) containing IS_STREET, ALTITUDE and IS_MODIFIABLE values,
    - output is a dictionary with keys 'is_street' and 'altitude', each being a numpy array of shape (clipping_size - input_surplus, clipping_size - input_surplus, 1).
    """
    # Define output signature for the dataset - helps TensorFlow understand the shape and type of the data
    assert input_third_dimension in [2, 3, 4], "input_third_dimension must be one of following values: [2, 3, 4]."
    y_spec_dict = {
        "is_street": tf.TensorSpec(
            shape=(clipping_size - input_surplus, clipping_size - input_surplus, 1), dtype=tf.float32
        )
    }
    if output_third_dimension >= 2:
        y_spec_dict["altitude"] = tf.TensorSpec(
            shape=(clipping_size - input_surplus, clipping_size - input_surplus, 1), dtype=tf.float32
        )
    if output_third_dimension >= 3:
        y_spec_dict["is_residential"] = tf.TensorSpec(
            shape=(clipping_size - input_surplus, clipping_size - input_surplus, 1), dtype=tf.float32
        )

    output_signature = (
        tf.TensorSpec(shape=(clipping_size, clipping_size, input_third_dimension), dtype=tf.float32),
        y_spec_dict,
    )

    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        lambda: clipping_sample_generator(
            files, cut_sizes, clipping_size, input_surplus, input_third_dimension, output_third_dimension
        ),
        output_signature=output_signature,
    )

    # Batch and prefetch the dataset - to improve performance
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


def clipping_sample_generator(
    grid_managers: list[GridManager],
    cut_sizes: list[tuple[int, int]],
    clipping_size: int,
    input_surplus: int,
    input_third_dimension: int,
    output_third_dimension: int,
):
    """Generates samples for clipping training.
    Yields tuples (input, output), where:
    - input is a numpy array of shape (clipping_size, clipping_size, 3) containing IS_STREET, ALTITUDE and IS_MODIFIABLE values,
    - output is a dictionary with keys 'is_street' and 'altitude', each being a numpy array of shape (clipping_size - input_surplus, clipping_size - input_surplus, 1).
    """
    assert input_third_dimension in [2, 3, 4], "input_third_dimension must be one of following values: [2, 3, 4]"
    assert output_third_dimension in [1, 2, 3], "output_third_dimension must be one of following values: [1, 2, 3]"
    while True:
        # Select a random file and cut size
        grid = random.choice(grid_managers)
        cut_size = random.choice(cut_sizes)

        # Get a random cut from the file
        _, cut = generate_cut(grid, cut_size)

        surplus_border = int(input_surplus / 2)

        # Determine random clipping position within the cut grid
        max_x = (
            cut.shape[1] - clipping_size + surplus_border
        )  # + (input_surplus / 2), because cut_from_cut subtracts clipping surplus
        max_y = cut.shape[0] - clipping_size + surplus_border

        clipping_x = random.randint(surplus_border, max_x)
        clipping_y = random.randint(surplus_border, max_y)

        # Create the clipping from the cut grid
        clipping = cut_from_cut(
            cut,
            clipping_x,
            clipping_y,
            (clipping_size, clipping_size),
            input_surplus,
            clipping=False,  # Otherwise we would use the indices of clippings and not coordinates -> we would end up taking mostly the bottom-right corner
        )

        # Prepare input and output for the model

        # Clean the clipping from IS_STREET data where IS_PREDICTED flag is on
        # cleaned_clipping = Model.clean_input(clipping, input_third_dimension)
        # without IS_RESIDENTIAL, but with IS_PREDICTED
        x = np.zeros((clipping.shape[0], clipping.shape[1], input_third_dimension), dtype=np.float32)
        # Fill IS_PREDICTED channel with ones -> we want to use all data for training

        if input_third_dimension >= 2:
            x[:, :, 1] = clipping[:, :, TRAINING_GRID_INDICES.IS_STREET].astype(np.float32)
        if input_third_dimension >= 3:
            x[:, :, 2] = clipping[:, :, TRAINING_GRID_INDICES.ALTITUDE].astype(np.float32)
        if input_third_dimension >= 4:
            x[:, :, 3] = clipping[:, :, TRAINING_GRID_INDICES.IS_RESIDENTIAL].astype(np.float32)

        crop = int(input_surplus / 2)

        # IMPORTANT: predict only the central area; keep border as known context
        # IS_PREDICTED = 1 in center, 0 on border
        x[:, :, TRAINING_GRID_INDICES.IS_PREDICTED] = 0.0
        x[crop : clipping_size - crop, crop : clipping_size - crop, TRAINING_GRID_INDICES.IS_PREDICTED] = 1.0

        # Now hide (zero) IS_STREET (and IS_RESIDENTIAL) only where IS_PREDICTED == 1
        x = Model.clean_input(x, input_third_dimension)

        # Prepare the area which we expect the model to predict
        output_clipping = clipping[
            crop : clipping_size - crop,
            crop : clipping_size - crop,
            1 : 1 + output_third_dimension,  # already without IS_PREDICTED
        ]

        # Prepare output values
        output = {
            "is_street": output_clipping[:, :, [PREDICT_GRID_INDICES.IS_STREET]].astype(
                np.float32
            )  # shape: (cut_size - input_surplus, cut_size - input_surplus, 1)
        }
        if output_third_dimension >= 2:
            output["altitude"] = output_clipping[:, :, [PREDICT_GRID_INDICES.ALTITUDE]].astype(
                np.float32
            )  # shape: (cut_size - input_surplus, cut_size - input_surplus, 1)
        if output_third_dimension >= 3:
            output["is_residential"] = output_clipping[:, :, [PREDICT_GRID_INDICES.IS_RESIDENTIAL]].astype(
                np.float32
            )  # shape: (cut_size - input_surplus, cut_size - input_surplus, 1)
        # Yield the input-output pair - TensorFlow will handle batching
        yield x, output


def generate_cut(grid_manager: GridManager[Grid], cut_size: tuple[int, int]) -> tuple[tuple[int, int], np.ndarray]:
    """Get the next random cut (part of a batch item) from the file (a batch item == multiple cuts)."""
    grid_metadata = grid_manager.get_metadata()
    grid_cols, grid_rows = grid_metadata.columns_number, grid_metadata.rows_number

    cut_w = min(cut_size[1], grid_cols)
    cut_h = min(cut_size[0], grid_rows)

    max_x = grid_cols - cut_w - 1  # max starting x for cut
    max_y = grid_rows - cut_h - 1  # max starting y for cut

    # Choose random starting point
    cut_start_x = random.randint(0, max_x) if max_x > 0 else 0
    cut_start_y = random.randint(0, max_y) if max_y > 0 else 0

    # Create cut grid
    cut = cut_from_grid_segments(grid_manager, cut_start_x, cut_start_y, (cut_h, cut_w), surplus=0, clipping=False)

    # append the IS_PREDICTED channel
    result = np.zeros((cut_h, cut_w, cut.shape[2] + 1), dtype=np.float32)
    result[:, :, 1:] = cut[:, :, :]

    # Return the cut starting position and the cut grid manager
    return ((cut_start_x, cut_start_y), result)
