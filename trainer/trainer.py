from dataclasses import dataclass

from trainer.clipping_model import ClippingModel


@dataclass
class Trainer:
    def __init__(self, model: ClippingModel, files: list[str]):
        """Initialize trainer with model and files.
        Args:
            model: Model to be trained.
            files (list[str]): A set of files to learn the model on."""
        self._model = model
        self._files = files

    def random_fit_from_files(self, epochs: int = 100, steps_per_epoch=1000, batch_size: int = 32):
        """Perform training on model.
        Args:
            fits_count (int): Number of fits to perform."""
        self._model.fit(
            train_files=self._files,
            val_files=self._files,
            cut_sizes=[(self._model.get_input_clipping_size(), self._model.get_input_clipping_size())],
            clipping_size=self._model.get_input_clipping_size(),
            input_surplus=self._model.get_input_grid_surplus(),
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )
        self._model.save()
