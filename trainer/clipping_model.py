import tensorflow as tf
Sequence = tf.keras.utils.Sequence

from CRoadA.trainer.model import Model
from CRoadA.trainer.clipping_sequence import ClippingBatchSequence

class ClippingModel(Model):
    def __init__(self):
        super().__init__()
        self._model = tf.keras.models.Sequential()
        # Model architecture here - #TODO: Define the actual architecture
        self._model.add(tf.keras.layers.InputLayer(input_shape=(None, None, 3)))
        self._model.add(tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'))
        self._model.add(tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'))
        self._model.add(tf.keras.layers.Conv2D(filters=2, kernel_size=1, activation='sigmoid', padding='same'))

    def fit(self, input: ClippingBatchSequence):
        """Fit model to the given data.

        Parameters
        ----------
        input : ClippingBatchSequence
            Input batch sequence with clipped grids.
        """
        # TODO: Implement proper training logic
        self._model.compile(optimizer='adam', loss='binary_crossentropy')
        self._model.fit(input, epochs=10)
    
    def predict(self, input):
        """Predicts grid for given input.

        Parameters
        ----------
        input : TODO

        Returns
        -------
        GridManager
            Predicted output grids.
        """
        predictions = self._model.predict(input)
        # Convert predictions to GridManager format if necessary
        return predictions