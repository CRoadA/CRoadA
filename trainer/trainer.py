from dataclasses import dataclass

from trainer.clipping_model import ClippingModel
import trainer.batch_sequence as batch_sequence
from trainer.clipping_sequence import ClippingBatchSequence


@dataclass
class Trainer:
    def __init__(self, model: ClippingModel, files: list[str]):
        """Initialize trainer with model and files.
        Args:
            model: Model to be trained.
            files (list[str]): A set of files to learn the model on."""
        self._model = model
        self._files = dict(zip(files, [0] * len(files)))  # file path -> count of uses in cutting

    def random_fit_from_files(self, epochs: int = 100, steps_per_epoch=1000):
        """Perform training on model.
        Args:
            fits_count (int): Number of fits to perform."""
        batchSeq = ClippingBatchSequence(
            batch_sequence.BatchSequence(
                files=list(self._files.keys()),
                batch_size=1,
                cut_sizes=[(self._model.get_input_clipping_size(), self._model.get_input_clipping_size())],
            ),
            clipping_size=self._model.get_input_clipping_size(),
            input_grid_surplus=self._model.get_input_grid_surplus(),
        )
        # TODO: Write proper references to proper training logic elements
        self._model.fit(batchSeq, epochs=epochs, steps_per_epoch=steps_per_epoch)
        self._model.save()
