from trainer.batch_sequence import *
from trainer.model import Model  # for clean_input()


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
        self._clipping_index = 0
        self._base_sequence = base_sequence

    def __len__(self) -> int:
        result = 0
        for base_batch_index in range(len(self._base_sequence)):
            cut_size = self._base_sequence.cut_sizes()[base_batch_index % len(self._base_sequence.cut_sizes())]
            rows_number, cols_number = (
                cut_size[0] - self._input_surplus,
                cut_size[1] - self._input_surplus,
            )
            result += (
                math.ceil(rows_number / (self._clipping_size - self._input_surplus))
                * math.ceil(cols_number / (self._clipping_size - self._input_surplus))
            ) * self._base_sequence.batch_size()
        return result

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:

        batches = None
        batch_item_index = 0
        clipping_index = 0
        result = 0
        found = False
        for base_batch_index in range(len(self._base_sequence)):
            for base_batch_item_index in range(self._base_sequence.batch_size()):
                cut_size = self._base_sequence.cut_sizes()[base_batch_index % len(self._base_sequence.cut_sizes())]
                rows_number, cols_number = (
                    cut_size[0] - self._input_surplus,
                    cut_size[1] - self._input_surplus,
                )
                result += math.ceil(rows_number / (self._clipping_size - self._input_surplus)) * math.ceil(
                    cols_number / (self._clipping_size - self._input_surplus)
                )
                if result >= index:
                    batches = self._base_sequence[base_batch_index]
                    batch_item_index = base_batch_item_index
                    clipping_index = index - (
                        result
                        - math.ceil(rows_number / (self._clipping_size - self._input_surplus))
                        * math.ceil(cols_number / (self._clipping_size - self._input_surplus))
                    )
                    found = True
                    break
            if found:
                break

        batch_x = []
        batch_y = []

        _, cut_grid = batches[batch_item_index]
        metadata = cut_grid.get_metadata()
        clipping_rows, clipping_cols = (
            math.ceil((metadata.rows_number - self._input_surplus) / (self._clipping_size - self._input_surplus)),
            math.ceil((metadata.columns_number - self._input_surplus) / (self._clipping_size - self._input_surplus)),
        )

        for _ in range(self._base_sequence.batch_size()):
            clipping_x = clipping_index % clipping_cols
            clipping_y = clipping_index // clipping_cols

            print("Clipping index:", clipping_index, "->", clipping_x, clipping_y)  # Debug print
            # Cut the clipping from the cut grid
            clipping = BatchSequence.cut_from_grid_segments(
                cut_grid,
                clipping_x,
                clipping_y,
                (self._clipping_size, self._clipping_size),
                self._input_surplus,
                clipping=True,
            )

            # TODO - when to clean? - (w uczeniu czasem powinien dostawać nie w pełni wyczyszczone dane czy nie?)
            clipping = Model.clean_input(clipping)
            batch_x.append(clipping[:, :, 0:2])  # TODO - without IS_RESIDENTIAL
            output_clipping = clipping[
                int(self._input_surplus / 2) : self._clipping_size - int(self._input_surplus / 2),
                int(self._input_surplus / 2) : self._clipping_size - int(self._input_surplus / 2),
                :,
            ]
            # TODO - adjust IS_REGIONAL value - probably we meant local, residential roads - IS_RESIDENTIAL value
            # batch_y.append(output_clipping[:, :, 0:2])  # only IS_STREET and ALTITUDE
            batch_y_is_street = output_clipping[:, :, 0:1]  # shape: (448, 448, 1)
            batch_y_altitude = output_clipping[:, :, 1:2]  # shape: (448, 448, 1)
            batch_y.append([batch_y_is_street, batch_y_altitude])

            print("Batch shapes so far:", np.stack(batch_x).shape, np.stack(batch_y).shape)  # Debug print

        batch_x = np.stack(batch_x)
        batch_y_is_street = np.stack([y[0] for y in batch_y])
        batch_y_altitude = np.stack([y[1] for y in batch_y])

        return batch_x, {"is_street": batch_y_is_street, "altitude": batch_y_altitude}
