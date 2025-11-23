class Trener():
    # model:

    def __init__(self, model):
        self.model = model

    def random_fit_from_files(self, files: list[str], fits_count: int = 100):
        """Perform training on model.
        Args:
            files (list[str]): A set of files to learn the model on.
            fits_count (int): Number of fits to perform."""