from dataclasses import dataclass

from CRoadA.trainer.model import Model
import CRoadA.trainer.batch_sequence as batch_sequence
from CRoadA.trainer.clipping_sequence import ClippingBatchSequence


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

        # batchSeq = data_types.BatchSequence(
        #     files=list(self._files.keys()),
        #     number_of_batches=1,
        #     batch_size=1,
        #     cut_size=[(32, 32), (64, 64), (128, 128)],
        # )
        batchSeq = ClippingBatchSequence(
            batch_sequence.BatchSequence(
                files=list(self._files.keys()),
                number_of_batches=1,
                batch_size=1,
                cut_size=[(256, 256)],
            ),
            clipping_size=256,
            input_grid_surplus=32,
        )
        # TODO
        # model.fit(batchSeq, epochs=fits_count) # pętla w modelu
