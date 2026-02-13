import numpy as np
from time import time
import os.path
from enum import Enum
import tensorflow as tf
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

from trainer.model import Model, TRAINING_GRID_INDICES, PREDICT_GRID_INDICES
from grid_manager import Grid, GridManager
from trainer.data_generator import InputGrid, OutputGrid, get_tf_dataset
from trainer.model_architectures import *
from trainer.model_metrics import _dice_coef, _dice_loss, FocalDiceLoss

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
    ClipModels.SHALLOWED_UNET: test_clipping_model_shallowed_unet,
}


class ClippingModel(Model):
    def __init__(
        self,
        model_type: ClipModels,
        clipping_size: int = 256,
        clipping_surplus: int = 64,
        input_third_dimension: int = 4,
        output_third_dimension: int = 3,
        weights: list[int] = [10, 1, 10],
        path: str | None = None,
        **kwargs,
    ):
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
        self._weights = weights

        def get_default_focal_dice_loss():
            return FocalDiceLoss(gamma=2.0, alpha=0.75, dice_weight=0.5)

        def _compile_model():
            # Prepare loss, loss weights and metrics dictionaries based on the output_third_dimension

            focal_dice = get_default_focal_dice_loss()

            loss = {"is_street": focal_dice}
            loss_weights = {"is_street": self._weights[0]}
            metrics = {
                "is_street": [
                    tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                    tf.keras.metrics.Precision(name="precision", thresholds=0.5),
                    tf.keras.metrics.Recall(name="recall", thresholds=0.5),
                    _dice_coef,
                ]
            }

            if self.output_third_dimension >= 2:
                loss["altitude"] = tf.keras.losses.Huber(delta=1.0)
                loss_weights["altitude"] = self._weights[1]
                metrics["altitude"] = [
                    tf.keras.metrics.MeanAbsoluteError(name="mae"),
                    tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                ]

            if self.output_third_dimension >= 3:
                loss["is_residential"] = focal_dice
                loss_weights["is_residential"] = self._weights[2]
                metrics["is_residential"] = [
                    tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                    tf.keras.metrics.Precision(name="precision", thresholds=0.5),
                    tf.keras.metrics.Recall(name="recall", thresholds=0.5),
                    _dice_coef,
                ]

            # Compile the model
            self._keras_model.compile(optimizer="adam", loss=loss, loss_weights=loss_weights, metrics=metrics)

        self.files = [f for f in os.listdir(self._dir) if os.path.isfile(os.path.join(self._dir, f))]
        if len(self.files) == 0:  # if no model exists in the path
            self._keras_model = clip_models[model_type](
                clipping_size=self._clipping_size,
                clipping_surplus=self._clipping_surplus,
                input_third_dimension=self.input_third_dimension,
                output_third_dimension=self.output_third_dimension,
                **kwargs,
            )
            _compile_model()
        else:
            self.files.sort(key=lambda file: file.split("_")[0])
            start_file = os.path.join(self._dir, self.files[-1])
            print(f"Starting from file: {start_file}")
            self._keras_model = tf.keras.models.load_model(
                start_file,
                custom_objects={
                    "FocalDiceLoss": get_default_focal_dice_loss(),
                    "_dice_coef": _dice_coef,
                    "_dice_loss": _dice_loss,
                },
            )

    def fit(
        self,
        train_files: list[GridManager[Grid]],
        val_files: list[GridManager[Grid]],
        cut_sizes: list[tuple[int, int]],
        clipping_size: int,
        input_surplus: int,
        batch_size: int,
        epochs: int = 1,
        steps_per_epoch: int = 1000,
    ):
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
        train_dataset = get_tf_dataset(
            train_files,
            cut_sizes,
            clipping_size,
            input_surplus,
            batch_size,
            input_third_dimension=self.input_third_dimension,
            output_third_dimension=self.output_third_dimension,
        )

        val_dataset = get_tf_dataset(
            val_files,
            cut_sizes,
            clipping_size,
            input_surplus,
            batch_size,
            input_third_dimension=self.input_third_dimension,
            output_third_dimension=self.output_third_dimension,
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_is_street__dice_coef" if self.output_third_dimension >= 2 else "val__dice_coef",
                mode="max",
                patience=20,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_is_street__dice_coef" if self.output_third_dimension >= 2 else "val__dice_coef",
                mode="max",
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    self._dir, str(int(time())) + "_epoch{epoch:02d}_val_loss{val_loss:.2f}" + "_model.keras"
                ),
                save_weights_only=False,
                save_best_only=False,
                verbose=1,
            ),
            tf.keras.callbacks.BackupAndRestore(backup_dir=os.path.join(self._dir, "backup")),
        ]

        # Fit the model using the datasets
        self._keras_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=steps_per_epoch // 10 if steps_per_epoch >= 10 else 1,
            callbacks=callbacks,
        )

    def predict(self, input: GridManager[InputGrid], debug_imgs: list[np.ndarray] = None) -> GridManager[OutputGrid]:
        """Predicts the output grid based on the input grid.

        :param input: Input grid manager containing the input grid for prediction.
        :type input: GridManager[InputGrid]
        :return: Output grid manager containing the predicted output grid.
        :rtype: GridManager[OutputGrid]
        """

        print(f"DEBUG: input.third_dimension_size: {input.get_metadata().third_dimension_size}")

        input_metadata = input.get_metadata()
        result_filename = f"from_{os.path.splitext(os.path.basename(input._file_name))[0]}__lat_{format(input_metadata.upper_left_latitude, '.4f').replace(".", "_")}__lon_{format(input_metadata.upper_left_longitude, '.4f').replace(".", "_")}__dim_{input_metadata.rows_number}x{input_metadata.columns_number}"
        result = GridManager(
            result_filename,
            input_metadata.rows_number - self._clipping_surplus,
            input_metadata.columns_number - self._clipping_surplus,
            0,
            0,
            input_metadata.grid_density,
            input_metadata.segment_h,
            input_metadata.segment_w,
            os.path.join("tmp", "predictions"),
            self.output_third_dimension,
        )
        result_metadata = result.get_metadata()
        result_h, result_w = result_metadata.rows_number, result_metadata.columns_number

        output_clipping_size = self._clipping_size - self._clipping_surplus
        feedback_third_dimension = min(self.input_third_dimension - 1, self.output_third_dimension)

        for row in range(0, result_h, output_clipping_size):
            print(f"\nrow: {row // output_clipping_size}", end="")

            # last row case handling
            row = min(row, result_h - output_clipping_size)

            if row > 0:
                already_predicted_context_height = min(
                    self._clipping_surplus // 2, row
                )  # it can happen, that the second row is already partial and there won't be a full clipping_surplus ready over it

            # Debug
            if debug_imgs is not None:
                debug_imgs.append([])

            for col in range(0, result_w, output_clipping_size):

                # last col case handling
                col = min(col, result_w - output_clipping_size)

                input_clipping = np.ones((self._clipping_size, self._clipping_size, self.input_third_dimension))
                input_clipping[:, :, :] = input.read_arbitrary_fragment(
                    row, col, self._clipping_size, self._clipping_size
                )[:, :, :]

                # Clean input
                # being_cleaned = input_clipping.copy()
                # being_cleaned[: self._clipping_surplus // 2, :, TRAINING_GRID_INDICES.IS_PREDICTED] = 0
                # being_cleaned[-self._clipping_surplus // 2 :, :, TRAINING_GRID_INDICES.IS_PREDICTED] = 0
                # being_cleaned[:, : self._clipping_surplus // 2, TRAINING_GRID_INDICES.IS_PREDICTED] = 0
                # being_cleaned[:, -self._clipping_surplus // 2 :, TRAINING_GRID_INDICES.IS_PREDICTED] = 0
                x = Model.clean_input(input_clipping, self.input_third_dimension)

                # Take already predicted values
                # Left Neighbor
                if col > 0:
                    already_predicted_context_width = min(
                        self._clipping_surplus // 2, col
                    )  # it can happen, that the second column is already partial and there won't be a full clipping_surplus ready left hand side of it
                    x[
                        self._clipping_surplus // 2 : -self._clipping_surplus // 2,
                        self._clipping_surplus // 2 - already_predicted_context_width : self._clipping_surplus // 2,
                        1 : feedback_third_dimension + 1,
                    ] = result.read_arbitrary_fragment(
                        row,
                        col - already_predicted_context_width,
                        output_clipping_size,
                        already_predicted_context_width,
                    )[
                        :, :, :feedback_third_dimension
                    ]

                # Top neighbors
                if row > 0:
                    if col > 0:
                        top_neighbors_width = output_clipping_size  + self._clipping_surplus
                        top_neighbors_offset = 0
                    else:
                        top_neighbors_width = output_clipping_size  + self._clipping_surplus // 2
                        top_neighbors_offset = self._clipping_surplus // 2
                    if col >= result_w - output_clipping_size - self._clipping_surplus // 2:
                        top_neighbors_width = result_w - (col - self._clipping_surplus // 2 + top_neighbors_offset)

                    already_predicted_context_height = min(
                        self._clipping_surplus // 2, row
                    )  # it can happen, that the second row is already partial and there won't be a full clipping_surplus ready over it
                    x[
                        :already_predicted_context_height,
                        top_neighbors_offset: top_neighbors_offset + top_neighbors_width,
                        1 : feedback_third_dimension + 1:
                    ] = result.read_arbitrary_fragment(
                        row - already_predicted_context_height,
                        col - self._clipping_surplus // 2 + top_neighbors_offset,
                        already_predicted_context_height,
                        top_neighbors_width,
                    )[
                        :, :, :feedback_third_dimension
                    ]

                    del top_neighbors_width
                    del top_neighbors_offset

                # Debug
                if debug_imgs is not None:
                    debug_imgs[-1].append(x)

                layers = list(self._keras_model(tf.expand_dims(x, axis=0)).values())

                output_clipping = np.zeros((output_clipping_size, output_clipping_size, self.output_third_dimension))
                for layer_i in range(len(layers)):
                    output_clipping[:, :, layer_i] = layers[layer_i][0, :, :, 0]

                output_clipping[..., PREDICT_GRID_INDICES.IS_STREET] = output_clipping[..., PREDICT_GRID_INDICES.IS_STREET] > 0.5
                if self.output_third_dimension >= 3:
                    output_clipping[..., PREDICT_GRID_INDICES.IS_RESIDENTIAL] = output_clipping[..., PREDICT_GRID_INDICES.IS_RESIDENTIAL] > 0.5

                result.write_arbitrary_fragment(output_clipping, row, col)
                print(".", end="")

        return result

    def assign_output_to_input(self, input_array: InputGrid, output_array: OutputGrid):
        third_dimension = min(self.input_third_dimension, self.output_third_dimension)
        input_array[:, :, 1 : third_dimension + 1] = output_array[:, :, :third_dimension]

    def assign_input_to_output(self, output_array: OutputGrid, input_array: InputGrid):
        third_dimension = min(self.input_third_dimension, self.output_third_dimension)
        output_array[:, :, :third_dimension] = input_array[:, :, 1 : third_dimension + 1]

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
