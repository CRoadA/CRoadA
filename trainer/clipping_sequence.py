from CRoadA.trainer.batch_sequence import *
from CRoadA.trainer.model import Model # for clean_input()

class ClippingBatchSequence(Sequence):
    """Batch sequence that clips input and output grids to proper sizes for the model."""

    def __init__(self, base_sequence: BatchSequence, clipping_size: int, input_grid_surplus: int):
        """Initializes clipping batch sequence.

        Parameters
        ----------
        base_sequence : BatchSequence
            Base batch sequence to clip data from.
        clipping_size : tuple[int, int]
            Size to clip the input and output grids to.
        input_grid_surplus : int
            Surplus size of input grid compared to output grid.
        """
        self._clipping_size = clipping_size
        self._input_surplus = input_grid_surplus
        self._batch_index = 0
        self._clipping_index = 0
        self._batches = list(base_sequence)

    def __len__(self) -> int:
        result = 0
        for batch in self._batches:
            start_point, cut = batch[0]
            metadata = cut.get_metadata()
            rows_number, cols_number = (
                metadata.rows_number - self._input_surplus,
                metadata.columns_number - self._input_surplus,
            )
            result += math.ceil(rows_number / (self._clipping_size - self._input_surplus)) * math.ceil(
                cols_number / (self._clipping_size - self._input_surplus)
            )

        return result

    def __getitem__(self, index: int) -> tuple[InputGrid, OutputGrid]:
        batch_x = []
        batch_y = []

        _, cut_grid = self._batches[self._batch_index][0]
        metadata = cut_grid.get_metadata()
        clipping_rows, clipping_cols = (
            math.ceil((metadata.rows_number - self._input_surplus) / (self._clipping_size - self._input_surplus)),
            math.ceil((metadata.columns_number - self._input_surplus) / (self._clipping_size - self._input_surplus)),
        )

        for _, cut_grid in self._batches[self._batch_index]:
            metadata = cut_grid.get_metadata()
            clipping_x = self._clipping_index % clipping_cols
            clipping_y = self._clipping_index // clipping_cols

            clipping_start_x = clipping_x * (self._clipping_size - self._input_surplus) - (self._input_surplus / 2)
            clipping_start_y = clipping_y * (self._clipping_size - self._input_surplus) - (self._input_surplus / 2)

            segment_h, segment_w = metadata.segment_h, metadata.segment_w

            which_segment_start_x = clipping_start_x // segment_h
            which_segment_start_y = clipping_start_y // segment_w

            which_segment_end_x = (clipping_start_x + self._clipping_size) // segment_h
            which_segment_end_y = (clipping_start_y + self._clipping_size) // segment_w

            clipping_x = np.array([])  # Grid()
            # Go by segments and merge them into one bigger cut - first vertically, then horizontally.
            for indx_x in range(which_segment_start_x, which_segment_end_x + 1):
                for indx_y in range(which_segment_start_y, which_segment_end_y + 1):
                    segment_y = cut_grid.read_segment(indx_x, indx_y)
                    # Merge segment_y into cut_y vertically.
                    if indx_y == which_segment_start_y:
                        clipping_y = segment_y
                    else:
                        clipping_y = np.vstack((clipping_y, segment_y))

                # Merge clipping_y into clipping_x horizontally.
                if indx_x == which_segment_start_x:
                    clipping_x = clipping_y
                else:
                    clipping_x = np.hstack((clipping_x, clipping_y))

            clipping_x = Model.clean_input(clipping_x)
            batch_x.append(clipping_x)
            output_clipping_x = clipping_x[
                int(self._input_surplus / 2) : self._clipping_size - int(self._input_surplus / 2),
                int(self._input_surplus / 2) : self._clipping_size - int(self._input_surplus / 2),
                :,
            ]
            # TODO - adjust IS_REGIONAL value
            batch_y.append(output_clipping_x[:, :, 0:2])  # only IS_STREET and ALTITUDE

        self._clipping_index += 1
        if clipping_rows * clipping_cols == self._clipping_index:
            self._clipping_index = 0
            self._batch_index += 1
            if self._batch_index >= len(self._batches):
                self._batch_index = 0

        return batch_x, batch_y