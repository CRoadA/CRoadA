import math
import tensorflow as tf

Sequence = tf.keras.utils.Sequence

from CRoadA.trainer.model import Model
from CRoadA.trainer.clipping_sequence import BatchSequence, ClippingBatchSequence, InputGrid, OutputGrid
from CRoadA.grid_manager import GridManager


class ClippingModel(Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.models.Sequential()
        # Model architecture here - #TODO: Define the actual architecture
        self.model.add(tf.keras.layers.InputLayer(input_shape=(None, None, 3)))
        self.model.add(tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"))
        self.model.add(tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"))
        self.model.add(tf.keras.layers.Conv2D(filters=2, kernel_size=1, activation="sigmoid", padding="same"))
        self.model.compile(optimizer="adam", loss="mse")

    def fit(self, input: ClippingBatchSequence, epochs: int = 1):
        """Fit model to the given data.

        Parameters
        ----------
        input : ClippingBatchSequence
            Input batch sequence with clipped grids.
        """
        # TODO: Implement proper training logic
        self.model.fit(input, epochs=epochs)

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
            clipping_size=256,
            input_grid_surplus=self.get_input_grid_surplus(),
        )
        for i in range(len(predict_sequence)):
            prediction = predict_sequence[i]
            # TODO - Combine predictions into a full grid manager?
            prediction_grid = BatchSequence.write_cut_to_grid_segments(
                prediction, 256, 256, 256, i * 256, i * 256, input._file_name, "./tmp/predictions/"
            )
            output_grids.append(prediction_grid)

        return output_grids  # This should be combined into a single GridManager

    def get_input_grid_surplus(self) -> int:
        """Get the surplus size of input grid compared to output grid.

        Returns
        -------
        int
            Number of rows/columns that the output grid is smaller than the input grid.
        """
        return 32  # Example surplus size


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

        self._grid_manager = grid_manager
        self._grid_metadata = self._grid_manager.get_metadata()
        self._grid_rows, self._grid_cols = self._grid_metadata.rows_number, self._grid_metadata.columns_number
        self._segment_rows, self._segment_cols = self._grid_metadata.segment_h, self._grid_metadata.segment_w

    def __len__(self):
        rows_number, cols_number = (
            self._grid_rows - self._input_surplus,
            self._grid_cols - self._input_surplus,
        )
        return math.ceil(rows_number / (self._clipping_size - self._input_surplus)) * math.ceil(
            cols_number / (self._clipping_size - self._input_surplus)
        )

    def __getitem__(self, index: int) -> GridManager[OutputGrid]:
        """
        Get one clipped prediction from the sequence.
        
        :param index: Index of the clipping to retrieve.
        :type index: int
        :return: Clipped prediction grid.
        :rtype: GridManager[OutputGrid]
        """
        cut_start_x = index * (self._clipping_size - self._input_surplus)
        cut_start_y = index * (self._clipping_size - self._input_surplus)
        batch_item = BatchSequence.cut_from_grid_segments(
            self._grid_manager,
            cut_start_x,
            cut_start_y,
            (self._clipping_size, self._clipping_size),
            surplus=self._input_surplus,
        )
        batch_item = Model.clean_input(batch_item)
        prediction = self._model.model.predict(tf.expand_dims(batch_item, axis=0))

        return prediction[0]
