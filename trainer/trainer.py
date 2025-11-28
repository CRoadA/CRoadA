from typing import Generator, Iterator
import numpy as np
import tensorflow as tf
Sequence = tf.keras.utils.Sequence
import numpy as np
from typing import Tuple

from grid_manager import Grid, GRID_INDICES, GridManager

class Trener():
    # model:

    def __init__(self, model, files: list[str]):
        """Initialize trainer with model and files.
        Args:
            model: Model to be trained.
            files (list[str]): A set of files to learn the model on."""
        self._model = model
        self._files = dict(zip(files, [0] * len(files))) # file path -> count of uses in cutting

    def random_fit_from_files(self, fits_count: int = 100):
        """Perform training on model.
        Args:
            fits_count (int): Number of fits to perform."""
        # TODO
    

    def _cut_random_file(self, cut_size: tuple[int, int]) -> Iterator[tuple[tuple[int, int], Grid], None, None]:
        """Cut segments from a random file.
        Args:
            cut_size (tuple[int, int]): Size of the cut to make (rows, columns) and then to pass to the model."""
        # TODO ?
        

    
    def cut_generator(self, cut_size: tuple[int, int]) -> Generator[tuple[tuple[int, int], Grid], None, None]:
        """TO BE DELETED - old version of cut random file - kept for reference and as a possible base for batch cutting."""
        import random
        file_path = random.choice(list(self._files.keys()))
        self._files[file_path] += 1 # increment count of uses

        grid_manager = GridManager(file_path) # load grid manager
        grid_metadata = grid_manager.get_metadata()
        grid_rows, grid_cols = grid_metadata.rows_number, grid_metadata.columns_number
        segment_rows, segment_cols = grid_metadata.segment_h, grid_metadata.segment_w
        # cut_size = self._model.input_size #  - get input size from the model TODO - rethink cut_size passing to the method
        max_x = grid_rows - cut_size[0] # max starting x for cut
        max_y = grid_cols - cut_size[1] # max starting y for cut
        
        already_used = set() # to avoid entirely overlapping cuts
        for i in range(max_x * max_y):
            start_x = random.randint(0, max_x)
            start_y = random.randint(0, max_y)
            while (start_x, start_y) in already_used: # avoid already used cuts
                start_x = random.randint(0, max_x)
                start_y = random.randint(0, max_y)
            already_used.add((start_x, start_y))
            end_x = start_x + cut_size[0]
            end_y = start_y + cut_size[1]

            which_segment_start_x = start_x // segment_rows
            which_segment_start_y = start_y // segment_cols
            which_segment_end_x = end_x // segment_rows
            which_segment_end_y = end_y // segment_cols

            # segments = []
            cut_x = Grid()
            """Go by segments and merge them into one bigger cut - first vertically, then horizontally."""
            for indx_x in range(which_segment_start_x, which_segment_end_x + 1):
                for indx_y in range(which_segment_start_y, which_segment_end_y + 1):
                    segment_y = grid_manager.read_segment(indx_x, indx_y)
                    # segments.append(((indx_x, indx_y), segment_y))
                    """Merge segment_y into cut_y vertically."""
                    if (indx_y == which_segment_start_y):
                        cut_y = segment_y
                    else:
                        cut_y = np.vstack((cut_y, segment_y))

                """Merge cut_y into cut_x horizontally."""
                if (indx_x == which_segment_start_x):
                    cut_x = cut_y
                else:
                    cut_x = np.hstack((cut_x, cut_y))
            
            yield ((start_x, start_y), cut_x)



class BatchSequence(Sequence):
    """Keras Sequence that yields a batch of cuts from possibly different files.
    Using Sequence improves performance by enabling the possibility of multi-threading and pre-fetching of data."""
    def __init__(self, trainer: Trener, number_of_batches: int, batch_size: int, cut_size: Tuple[int, int]):
        self._files = trainer._files
        self._number_of_batches = number_of_batches
        self._batch_size = batch_size
        self._iterators = [CutIterator(cut_size=cut_size, sequence=self) for _ in range(batch_size) ] # pre-create iterators // previous version was creating new iterator for each batch item, therefore not necessarily the same cut-size for each item in a batch
        # I was thinking about making each iterator correspond to a specific part (segments) of each file
        # OR about reading firstly -> random parts of one file
    
    def __len__(self):
        # Number of batches per epoch
        return self._number_of_batches

    def __getitem__(self, cut_size: tuple[int, int]):
        # Generate one batch of data
        batches = []
        for iterator in self._iterators:
            batch = []
            for _ in range(self._batch_size):
                batch.append(next(iterator))
            batches.append(batch)
        return batches
    
    def on_epoch_end(self):
        # Optional method e.g. to shuffle data at the end of each epoch
        pass



class CutIterator:
    """Iterator that yields random cuts from a random file in a BatchSequence."""
    def __init__(self, cut_size: tuple[int, int], sequence: BatchSequence):
        import random
        self._file_path = random.choice(list(sequence._files.keys()))
        sequence._files[self._file_path] += 1 # increment count of uses
        self._cut_size = cut_size

        self._grid_manager = GridManager(self._file_path) # load grid manager
        self._grid_metadata = self._grid_manager.get_metadata()
        self._grid_rows, self._grid_cols = self._grid_metadata.rows_number, self._grid_metadata.columns_number
        self._segment_rows, self._segment_cols = self._grid_metadata.segment_h, self._grid_metadata.segment_w

        self._max_x = self._grid_rows - cut_size[0] # max starting x for cut
        self._max_y = self._grid_cols - cut_size[1] # max starting y for cut
        self._already_used = set() # to avoid entirely overlapping cuts
    
    def __iter__(self) -> Iterator[tuple[tuple[int, int], Grid], None, None]:
        return self
    
    def __next__(self) -> tuple[tuple[int, int], Grid]:
        import random
        for _ in range(self._max_x * self._max_y):
            start_x = random.randint(0, self._max_x)
            start_y = random.randint(0, self._max_y)
            while (start_x, start_y) in self._already_used: # avoid already used cuts
                start_x = random.randint(0, self._max_x)
                start_y = random.randint(0, self._max_y)
            self._already_used.add((start_x, start_y))
            end_x = start_x + self._cut_size[0]
            end_y = start_y + self._cut_size[1]

            which_segment_start_x = start_x // self._segment_rows
            which_segment_start_y = start_y // self._segment_cols
            which_segment_end_x = end_x // self._segment_rows
            which_segment_end_y = end_y // self._segment_cols

            cut_x = Grid()
            """Go by segments and merge them into one bigger cut - first vertically, then horizontally."""
            for indx_x in range(which_segment_start_x, which_segment_end_x + 1):
                for indx_y in range(which_segment_start_y, which_segment_end_y + 1):
                    segment_y = self._grid_manager.read_segment(indx_x, indx_y)
                    """Merge segment_y into cut_y vertically."""
                    if (indx_y == which_segment_start_y):
                        cut_y = segment_y
                    else:
                        cut_y = np.vstack((cut_y, segment_y))

                """Merge cut_y into cut_x horizontally."""
                if (indx_x == which_segment_start_x):
                    cut_x = cut_y
                else:
                    cut_x = np.hstack((cut_x, cut_y))
            
            return ((start_x, start_y), cut_x)
        
        raise StopIteration