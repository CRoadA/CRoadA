import math
from time import time
import numpy as np
import tensorflow as tf
import os.path
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

Sequence = tf.keras.utils.Sequence

from trainer.model import Model
from grid_manager import GridManager
from trainer.data_generator import InputGrid, OutputGrid, get_tf_dataset
from trainer.cut_grid import cut_from_grid_segments, write_cut_to_grid_segments


class ClippingModel(Model):
    def __init__(self, clipping_size: int = 512, clipping_surplus: int = 64, path: str | None = None):
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

        if not path is None and not os.path.isfile(path):
            # Model architecture here
            inputs = tf.keras.layers.Input(shape=(self._clipping_size, self._clipping_size, 2)) # TODO - without IS_RESIDENTIAL
            x = tf.keras.layers.Conv2D(16, 5, activation="relu", padding="same", strides=1)(inputs)
            x = tf.keras.layers.Conv2D(8, 5, activation="relu", padding="same", strides=1)(x)
            x = tf.keras.layers.Conv2D(4, 5, activation="relu", padding="same", strides=1)(x)

            # How much to crop to get rid of surplus
            crop = self._clipping_surplus // 2
            x = tf.keras.layers.Cropping2D(cropping=((crop, crop), (crop, crop)))(x)

            # One output layer with two channels
            x = tf.keras.layers.Conv2D(2, 1, activation=None, name="output")(x)

            # Split outputs into two separate heads
            out_is_street = tf.keras.layers.Lambda(lambda t: tf.keras.activations.sigmoid(t[..., 0:1]), name="is_street", dtype="float32")(x)
            out_altitude = tf.keras.layers.Lambda(lambda t: t[..., 1:2], name="altitude", dtype="float32")(x)
            outputs = [out_is_street, out_altitude]

            self._keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)


            # self._keras_model.compile(
            #     optimizer="adam",
            #     loss="mse",
            #     loss_weights=None,
            #     metrics=None,
            #     weighted_metrics=None,
            #     run_eagerly=False,
            #     steps_per_execution=1,
            #     jit_compile="auto",
            #     auto_scale_loss=True,
            # )
            self._keras_model.compile(
                optimizer="adam",
                loss={
                    "is_street": "binary_crossentropy",
                    "altitude": "mse"
                },
                loss_weights={"is_street": 1.0, "altitude": 1.0},
                metrics={
                    "is_street": ["accuracy"],
                    "altitude": ["mae"]
                }
            )
        else:
            self._keras_model = tf.keras.models.load_model(path)

    def fit(self, train_files: list[str], val_files: list[str], cut_sizes: list[tuple[int, int]], clipping_size: int, input_surplus: int, batch_size: int, epochs: int = 1, steps_per_epoch: int = 1000):
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
        train_dataset = get_tf_dataset(train_files, cut_sizes, clipping_size, input_surplus, batch_size)
        val_dataset = get_tf_dataset(val_files, cut_sizes, clipping_size, input_surplus, batch_size)
        # Fit the model using the datasets
        self._keras_model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=steps_per_epoch // 10)

    def predict(self, input: GridManager[InputGrid]) -> list[GridManager[OutputGrid]]:
        """Predicts grid for given input.

        Parameters
        ----------
        input : GridManager[InputGrid]
            Input grid manager with input grids.

        Returns
        -------
        list[GridManager[OutputGrid]]
            Predicted output grids.
        """
        output_grids = []
        predict_sequence = PredictClippingSequence(
            model=self,
            grid_manager=input,
            clipping_size=self._clipping_size,
            input_grid_surplus=self.get_input_grid_surplus(),
        )
        for i in range(len(predict_sequence)):
            prediction = predict_sequence[i]
            # TODO - Combine predictions into a full grid manager?
            prediction_grid = write_cut_to_grid_segments(
                prediction, self._clipping_size, self._clipping_size, self._clipping_size,
                i * self._clipping_size, i * self._clipping_size, input._file_name, "./tmp/predictions/"
            )  # TODO - check if the name is sufficient
            output_grids.append(prediction_grid)

        return output_grids  # This should be combined into a single GridManager

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


class PredictClippingSequence(Sequence):
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


def unet(input_shape=(256, 256, 3), n_classes=1):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    c1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    # Bottleneck
    b = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
    b = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(b)
    # Decoder
    u3 = tf.keras.layers.UpSampling2D()(b)
    u3 = tf.keras.layers.concatenate([u3, c3])
    c4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(u3)
    c4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c4)

    u2 = tf.keras.layers.UpSampling2D()(c4)
    u2 = tf.keras.layers.concatenate([u2, c2])
    c5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
    c5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c5)

    u1 = tf.keras.layers.UpSampling2D()(c5)
    u1 = tf.keras.layers.concatenate([u1, c1])
    c6 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(u1)
    c6 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(c6)

    # Output heads
    street_out = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', name='is_street')(c6)
    altitude_out = tf.keras.layers.Conv2D(1, 1, activation='linear', name='altitude')(c6)

    model = tf.keras.Model(inputs, [street_out, altitude_out])
    return model