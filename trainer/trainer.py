from dataclasses import dataclass

from trainer.model import Model
import trainer.batch_sequence as batch_sequence
from trainer.clipping_sequence import ClippingBatchSequence


@dataclass
class Trainer:
    def __init__(self, model: Model, files: list[str]):
        """Initialize trainer with model and files.
        Args:
            model: Model to be trained.
            files (list[str]): A set of files to learn the model on."""
        self._model = model
        self._files = dict(zip(files, [0] * len(files)))  # file path -> count of uses in cutting

    def random_fit_from_files(self, fits_count: int = 100):
        """Perform training on model.
        Args:
            fits_count (int): Number of fits to perform."""
        batchSeq = ClippingBatchSequence(
            batch_sequence.BatchSequence(
                files=list(self._files.keys()),
                batch_size=1,
                cut_size=[(256, 256)],
            ),
            clipping_size=256,
            input_grid_surplus=self._model.get_input_grid_surplus(),
        )
        # TODO: Write proper references to proper training logic elements
        self._model.fit(batchSeq, epochs=fits_count)
        self._model.save()
