import math
import numpy as np
from time import time
import os.path
from enum import Enum
import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

Sequence = tf.keras.utils.Sequence

from trainer.model import Model, GRID_INDICES
from grid_manager import Grid, GridManager
from trainer.data_generator import InputGrid, OutputGrid, get_tf_dataset
from trainer.cut_grid import cut_from_grid_segments, write_cut_to_grid_segments
from trainer.model_architectures import *

THIRD_DIMENSION = 3  # IS_STREET, ALTITUDE, IS_MODIFIABLE

class ClipModels(Enum):
    BASE = "base_clipping_model"
    UNET = "unet"
    ALEX_INSPIRED = "alex_inspired"
    SHALLOWED_UNET = "shallowed_unet"

clip_models = {
    ClipModels.BASE: base_clipping_model,
    ClipModels.UNET: unet,
    ClipModels.ALEX_INSPIRED: alex_inspired,
    ClipModels.SHALLOWED_UNET: test_clipping_model_shallowed_unet
}

class ClippingModel(Model):
    def __init__(self, model_type: ClipModels, clipping_size: int = 512, clipping_surplus: int = 64, input_third_dimension: int = 3, output_third_dimension: int = 2, weights: list[int] = [10, 1, 10], path: str | None = None, **kwargs):
        """
        Initializes the ClippingModel with specified clipping size and surplus.
        
        :param clipping_size: Size of the clipping for input grids.
        :type clipping_size: int
        :param clipping_surplus: Surplus size of the input grid compared to the output grid; output grid will be smaller than clipping_size by this amount.
        :type clipping_surplus: int
        :param path: Path to a saved model file.
        :type path: str | None
        """
        super().__init__(path)
        self._clipping_size = clipping_size
        self._clipping_surplus = clipping_surplus
        self.input_third_dimension = input_third_dimension
        self.output_third_dimension = output_third_dimension

        files = [f for f in os.listdir(self._dir) if os.path.isfile(os.path.join(self._dir, f))]
        if len(files) == 0: # if no model exists in the path
            # Model architecture here
            self._keras_model = clip_models[model_type](clipping_size=self._clipping_size, clipping_surplus=self._clipping_surplus, input_third_dimension=self.input_third_dimension, output_third_dimension=self.output_third_dimension, **kwargs) # TODO - without IS_RESIDENTIAL (third dimension = 3)

            # Compile the model
            loss = {
                "is_street": "binary_crossentropy"
            }
            loss_weights = {
                "is_street": weights[0]
            }
            metrics = {
                "is_street": ["accuracy"]
            }

            if output_third_dimension >= 2:
                loss["altitude"] = "mse"
                loss_weights["altitude"] = weights[1]
                metrics["altitude"] = "mae"
            if output_third_dimension >= 3:
                loss["is_residential"] = "binary_crossentropy"
                loss_weights["is_residential"] = weights[2]
                metrics["is_residential"] = "accuracy"

            self._keras_model.compile(
                optimizer="adam",
                loss=loss,
                loss_weights=loss_weights,
                metrics=metrics
            )
        else:
            files.sort(key=lambda file: file.split("_")[0])
            start_file = os.path.join(self._dir, files[-1])
            print(f"Starting from file: {start_file}")
            self._keras_model = tf.keras.models.load_model(start_file)

    def fit(self, train_files: list[GridManager[Grid]], val_files: list[GridManager[Grid]], cut_sizes: list[tuple[int, int]], clipping_size: int, input_surplus: int, batch_size: int, epochs: int = 1, steps_per_epoch: int = 1000):
        """Fit model to the given data.

        Parameters
        ----------
        train_files : list[str]
            Input batch sequence with clipped grids.
        val_files : list[str]
            Validation batch sequence with clipped grids.
        cut_sizes : list[tuple[int, int]]
            List of cut sizes to use for generating training data.
        clipping_size : int
            Size of the clipping for input grids.
        input_surplus : int
            Surplus size of input grid compared to output grid.
        batch_size : int
            Size of each training batch.
        epochs : int, optional
            Number of epochs to train the model, by default 1
        steps_per_epoch : int, optional
            Number of steps per epoch, by default 1000
        """
        # Create TensorFlow datasets for training and validation
        train_dataset = get_tf_dataset(train_files, cut_sizes, clipping_size, input_surplus, batch_size, input_third_dimension=self.input_third_dimension, output_third_dimension=self.output_third_dimension)

        val_dataset = get_tf_dataset(val_files, cut_sizes, clipping_size, input_surplus, batch_size, input_third_dimension=self.input_third_dimension, output_third_dimension=self.output_third_dimension)
        # Fit the model using the datasets
        self._keras_model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=steps_per_epoch // 10 if steps_per_epoch >= 10 else 1)

    def predict(self, input: GridManager[InputGrid]) -> GridManager[OutputGrid]:
        
        input_metadata = input.get_metadata()
        result_filename = f"from_{os.path.splitext(os.path.basename(input._file_name))[0]}__lat_{format(input_metadata.upper_left_latitude, ".4f").replace(".", "_")}__lon_{format(input_metadata.upper_left_longitude, ".4f").replace(".", "_")}__dim_{input_metadata.rows_number}x{input_metadata.columns_number}"
        result = GridManager(
            result_filename,
            input_metadata.rows_number - self._clipping_surplus,
            input_metadata.columns_number - self._clipping_surplus,
            0,0,
            input_metadata.grid_density,
            input_metadata.segment_h,
            input_metadata.segment_w,
            os.path.join("tmp", "predictions"),
            self.output_third_dimension
        )
        result_metadata = result.get_metadata()
        result_h, result_w = result_metadata.rows_number, result_metadata.columns_number

        left_neighbor = None
        output_clipping_size = self._clipping_size - self._clipping_surplus

        top_neighbors = np.zeros((output_clipping_size, 3 * output_clipping_size))
        for row in range(0, result_h, output_clipping_size):

            # last row case handling
            row = min(row, result_h - output_clipping_size)

            if row > 0:
                top_neighbors[:, output_clipping_size:] = result.read_arbitrary_fragment(row - output_clipping_size, 0, output_clipping_size, 2 * output_clipping_size)

            for col in range(0, result_w, output_clipping_size):

                # last col case handling
                col = min(col, result_w - output_clipping_size)

                input_clipping = np.ones(self._clipping_size, self._clipping_size, self.input_third_dimension)
                input_clipping[:, :, 1:self.input_third_dimension] = input.read_arbitrary_fragment(
                    row,
                    col,
                    self._clipping_size,
                    self._clipping_size
                )[:self.input_third_dimension-1]
                
                # Take already pedicted values
                if row > 0:
                    if col > 0:
                        input_clipping[
                            :self._clipping_surplus,
                            :,
                            1:self.output_third_dimension + 1
                        ] = top_neighbors[
                            :self._clipping_surplus,
                            output_clipping_size - self._clipping_surplus : 2*output_clipping_size + self._clipping_surplus,
                            :
                        ]
                    else:
                        input_clipping[
                            :self._clipping_surplus,
                            self._clipping_surplus:,
                            1:self.output_third_dimension + 1
                        ] = top_neighbors[
                            :self._clipping_surplus,
                            output_clipping_size : 2*output_clipping_size + self._clipping_surplus,
                            :
                        ]

                if col > 0:
                    input_clipping[:, :self._clipping_surplus, 1:self.output_third_dimension + 1] = left_neighbor[:, -self._clipping_surplus:, :]

                # Clean input
                x = Model.clean_input(input_clipping[0: self.input_third_dimension])
                #Predict
                output_clipping = self._model._keras_model.predict(tf.expand_dims(x, axis=0))
                result.write_arbitrary_fragment(output_clipping, row, col)

                # Update neighbors
                left_neighbor = output_clipping
                if row > 0:
                    top_neighbors[
                        :,
                        :2*output_clipping_size,
                        :
                    ] = top_neighbors[
                        :,
                        output_clipping_size:,
                        :
                    ]
                    top_neighbors[
                        :,
                        2*output_clipping_size,
                        :
                    ] = result.read_arbitrary_fragment(row - output_clipping_size, col + 2*output_clipping_size, output_clipping_size, output_clipping_size)

        return result



    # def predict(self, input: GridManager[InputGrid]) -> list[GridManager[OutputGrid]]:
    #     """Predicts grid for given input.

    #     Parameters
    #     ----------
    #     input : GridManager[InputGrid]
    #         Input grid manager with input grids.

    #     Returns
    #     -------
    #     list[GridManager[OutputGrid]]
    #         Predicted output grids.
    #     """
    #     output_grids = []
    #     predict_sequence = PredictClippingSequence(
    #         model=self,
    #         grid_manager=input,
    #         clipping_size=self._clipping_size,
    #         input_grid_surplus=self.get_input_grid_surplus(),
    #     )
    #     for i in range(len(predict_sequence)):
    #         prediction = predict_sequence[i]
    #         # TODO - Combine predictions into a full grid manager?
    #         prediction_grid = write_cut_to_grid_segments(
    #             prediction, self._clipping_size, self._clipping_size, self._clipping_size,
    #             i * self._clipping_size, i * self._clipping_size, input._file_name, "./tmp/predictions/"
    #         )  # TODO - check if the name is sufficient
    #         output_grids.append(prediction_grid)

    #     return output_grids  # This should be combined into a single GridManager

    def get_input_grid_surplus(self) -> int:
        """Get the surplus size of input grid compared to output grid.

        Returns
        -------
        int
            Number of rows/columns that the output grid is smaller than the input grid.
        """
        return self._clipping_surplus
    
    def get_input_clipping_size(self) -> int:
        """Get the clipping size used for input grids.

        Returns
        -------
        int
            Clipping size.
        """
        return self._clipping_size
    
    def save(self):
        """Saves the model to the specified path.

        Parameters
        ----------
        path : str
            Path to save the model.
        """
        if not os.path.exists(self._dir):
            os.makedirs(self._dir)
        self._keras_model.save(os.path.join(self._dir, str(int(time()))) + "_model.keras")


class PredictClippingSequence(Sequence): # TODO - test and modify if needed
    """Sequence that divides input grids into batches for prediction using ClippingModel."""

    def __init__(
        self, model: ClippingModel, grid_manager: GridManager[InputGrid], clipping_size: int, input_grid_surplus: int
    ):
        """
        Initializes the PredictClippingSequence used for making predictions with ClippingModel.
        
        :param model: ClippingModel instance for making predictions.
        :type model: ClippingModel
        :param grid_manager: GridManager containing input grids.
        :type grid_manager: GridManager[InputGrid]
        :param clipping_size: Size of the clipping for prediction.
        :type clipping_size: int
        :param input_grid_surplus: Surplus size of input grid compared to output grid.
        :type input_grid_surplus: int
        """
        self._model = model
        self._clipping_size = clipping_size
        self._input_surplus = input_grid_surplus

        self._grid_manager = grid_manager.deep_copy()
        self._grid_metadata = self._grid_manager.get_metadata()
        self._grid_rows, self._grid_cols = self._grid_metadata.rows_number, self._grid_metadata.columns_number
        self._segment_rows, self._segment_cols = self._grid_metadata.segment_h, self._grid_metadata.segment_w

    def __len__(self) -> int:
        rows_number, cols_number = (
            self._grid_rows - self._input_surplus,
            self._grid_cols - self._input_surplus,
        )
        return math.ceil(rows_number / (self._clipping_size - self._input_surplus)) * math.ceil(
            cols_number / (self._clipping_size - self._input_surplus)
        )

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Get one clipped prediction from the sequence.
        
        :param index: Index of the clipping to retrieve.
        :type index: int
        :return: Clipped prediction grid.
        :rtype: np.ndarray
        """
        # Calculate clipping start positions - cover the whole grid, moving by clipping size minus surplus - not to overlap and simultaneously cover all areas
        step = self._clipping_size - self._input_surplus
        n_cols = math.ceil((self._grid_cols - self._input_surplus) / step)
        row = index // n_cols
        col = index % n_cols
        cut_start_x = (
            col * step
        )  # TODO - what when the surplus makes us go out of bounds - should we complete the grid with zeros?
        cut_start_y = row * step

        batch_item = cut_from_grid_segments(
            self._grid_manager,
            cut_start_x,
            cut_start_y,
            (self._clipping_size, self._clipping_size),
            surplus=self._input_surplus,
        )
        batch_item = Model.clean_input(batch_item)
        prediction = self._model._keras_model.predict(tf.expand_dims(batch_item, axis=0))
        #self._grid_manager.write_segment(prediction[0], cut_start_y, cut_start_x) # TODO - what if clippings are smaller than segment size?
        return prediction[0]
