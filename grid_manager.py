import pickle
import os
import numpy as np
from typing import Any, Literal, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
import math
import sys
import copy
import shutil
import time


class GRID_INDICES:
    IS_STREET = 0
    ALTITUDE = 1
    IS_RESIDENTIAL = 2


METADATA_SEPARATOR = ";"

Grid = np.ndarray[(Any, Any, 3), Any]  # point on Grid (row, col, value_index)
"""Type of numpy array with points grid (row, col, value_index). Value_index should be taken from GRID_INDICES."""


@dataclass
class GridFileMetadata:
    """Metadata of grid file.
    Attributes:
        version (int): Format version idenifier. An appropriate version of read/write function must be used (or just a general-purpose one).
        rows_number (int): Number of rows of the whole grid (not just the segment).
        columns_number (int): Number of columns of the whole grid (not just the segment).
        upper_left_longitude (float): Longitude of upper-left-most point in the file.
        upper_left_latitude (float): Latitude of upper-left-most point in the file.
        grid_density (float): Distance between two distinct closest points in grid (in meters).
        segment_h (int): Height of segment (in grid rows).
        segment_w (int): Width of segment (in grid columns).
        byteorder ("little"|"big"): Big-endian or little-endian.
        third_dimension_size (int): Size of the third dimension of the array.
        metadata_bytes (int): Length of the metadata line (in bytes)."""

    version: int
    rows_number: int
    columns_number: int
    upper_left_longitude: float
    upper_left_latitude: float
    grid_density: float
    segment_h: int
    segment_w: int
    byteorder: Literal["little", "big"]
    third_dimension_size: int
    metadata_bytes: int


GridType = TypeVar("GridType", bound=np.ndarray)

SINGLE_CELL_SIZE = 4


class GridManager(Generic[GridType]):
    _file_name: str
    _data_dir: str

    _metadata: GridFileMetadata

    def __init__(
        self,
        file_name: str,
        rows_number: int | None = None,
        columns_number: int | None = None,
        upper_left_longitude: float | None = None,
        upper_left_latitude: float | None = None,
        grid_density: float | None = 1,
        segment_h: int = 5000,
        segment_w: int = 5000,
        data_dir: str = "grids",
        third_dimension_size: int = 3,
    ):
        """Create GridManager, which manages reading and writing to a specific grid file.
        Args:
            file_name (str): File name.
            rows_number (int): Number of rows of the whole gird (not just the segment).
                - None: Value is read from metadata of existing file or the FileNotFoundError is raised if such file does not exist.
                - int: The file is created with such metadata or FileExistsError is raised when file already exists and the metadata parameter does not match.
            columns_number (int): Number of columns of the whole gird (not just the segment). None and int values are handled as by rows_number.
            upper_left_longitude (float): Longitude of the upper-left-most point.
            upper_left_latitude (float): Latitude of the upper-left-most point.
            grid_density (float): Distance between two distinct closest points in grid (in meters).
            segment_h (int): Height of segment (in grid rows).
            segment_w (int): Width of segment (in grid columns).
            third_dimension_size (int): Size of the third dimension of the array.
            data_dir (str): Folder with the file.
        """
        self._file_name = file_name
        self._data_dir = data_dir

        if not os.path.exists(self._data_dir):
            os.makedirs(self._data_dir)

        self._create_file(
            rows_number,
            columns_number,
            upper_left_longitude,
            upper_left_latitude,
            grid_density,
            segment_h,
            segment_w,
            third_dimension_size,
        )
        self._metadata = self._read_metadata()

    def write_segment(self, segment: GridType, segment_row: int, segment_col: int):
        """Write segment of grid to a file. General-purpose, file-format-agnostic function.
        Args:
            segment (GridType): Grid segment.
            segment_row (int): 0-based row index of the segment (in terms of segments, not the grids columns).
            segment_col (int): 0-based column index of the segment (in terms of segments, not the grids columns).
        """

        if self._metadata.version == 1:
            self._write_segment_v1(segment, segment_row, segment_col)
        else:
            raise ValueError(f"Unsupported file version {self._metadata.version}")

    def read_segment(self, segment_row: int, segment_col: int) -> GridType:
        """Read segment from given file.
        Args:
            segment_row (int): 0-based row index of the segment (in terms of segments, not the grids columns).
            segment_col (int): 0-based column index of the segment (in terms of segments, not the grids columns)."""

        if self._metadata.version == 1:
            return self._read_segment_v1(segment_row, segment_col)
        else:
            raise ValueError(f"Unsupported file version {self._metadata.version}")

    # Corrected by ChatGPT:
    # CHANGE: new general-purpose functions for reading/writing arbitrary fragments of the grid (not just whole segments). These functions read/write only required bytes from/to the file, thus they are more efficient when the fragment is smaller than the segment.
    def read_arbitrary_fragment(self, row: int, col: int, height: int, width: int) -> GridType:
        assert (
            row + height <= self._metadata.rows_number
        ), f"row + height cannot exceed number of rows of grid manager. Got row: {row}, height: {height}, while file's rows number is {self._metadata.rows_number}"
        assert (
            col + width <= self._metadata.columns_number
        ), f"col + width cannot exceed number of columns of grid manager. Got col: {col}, width: {width}, while file's colums number is {self._metadata.columns_number}"

        metadata = self._metadata
        segment_h, segment_w = metadata.segment_h, metadata.segment_w

        start_segment_row = row // segment_h
        start_segment_column = col // segment_w
        end_segment_row = (row + height - 1) // segment_h
        end_segment_column = (col + width - 1) // segment_w

        third = metadata.third_dimension_size
        result = np.zeros((height, width, third), dtype=np.float32)

        # CHANGE: read only required bytes instead of whole segments
        file_path = os.path.join(self._data_dir, self._file_name)
        with open(file_path, "rb", buffering=0) as f:
            fd = f.fileno()

            rows_n = metadata.rows_number
            cols_n = metadata.columns_number
            segments_n_vertically = math.ceil(rows_n / segment_h)
            segments_n_horizontally = math.ceil(cols_n / segment_w)

            for segment_row in range(start_segment_row, end_segment_row + 1):
                for segment_col in range(start_segment_column, end_segment_column + 1):
                    start_row = max(segment_row * segment_h, row)
                    start_col = max(segment_col * segment_w, col)

                    end_row = min((segment_row + 1) * segment_h, row + height)
                    end_col = min((segment_col + 1) * segment_w, col + width)

                    # local coords inside segment
                    local_y0 = start_row - segment_row * segment_h
                    local_y1 = end_row - segment_row * segment_h
                    local_x0 = start_col - segment_col * segment_w
                    local_x1 = end_col - segment_col * segment_w

                    local_h = local_y1 - local_y0
                    local_w = local_x1 - local_x0
                    if local_h <= 0 or local_w <= 0:
                        continue

                    # actual stored segment shape (last row/col can be smaller)
                    seg_h_cur = segment_h
                    if segment_row == segments_n_vertically - 1:
                        seg_h_cur = rows_n % segment_h or segment_h

                    seg_w_cur = segment_w
                    if segment_col == segments_n_horizontally - 1:
                        seg_w_cur = cols_n % segment_w or segment_w

                    # base byte offset of the segment in file
                    seg_base = self._coords_to_file_position(segment_row, segment_col)

                    # read exact rectangle row-by-row (minimizes disk bytes)
                    bytes_per_row = local_w * third * SINGLE_CELL_SIZE
                    out_block = np.empty((local_h, local_w, third), dtype=np.float32)

                    for i in range(local_h):
                        seg_r = local_y0 + i
                        # offset inside segment (row-major within segment)
                        byte_off = seg_base + ((seg_r * seg_w_cur + local_x0) * third * SINGLE_CELL_SIZE)
                        chunk = os.pread(fd, bytes_per_row, byte_off)
                        if len(chunk) != bytes_per_row:
                            raise IOError(
                                f"Short read while reading fragment: got {len(chunk)} bytes, expected {bytes_per_row}"
                            )
                        out_block[i, :, :] = np.frombuffer(chunk, dtype=np.float32).reshape(local_w, third)

                    # write into output
                    result[start_row - row : end_row - row, start_col - col : end_col - col, :] = out_block

        return result

    def write_arbitrary_fragment_fast(self, fragment: GridType, row: int, col: int) -> None:
        """Faster variant: writes only the required bytes (no full-segment read/overwrite)."""
        height, width = fragment.shape[0], fragment.shape[1]
        assert height > 0 and width > 0
        assert row + height <= self._metadata.rows_number
        assert col + width <= self._metadata.columns_number

        metadata = self._metadata
        segment_h, segment_w = metadata.segment_h, metadata.segment_w
        third = metadata.third_dimension_size

        start_segment_row = row // segment_h
        start_segment_column = col // segment_w
        end_segment_row = (row + height - 1) // segment_h
        end_segment_column = (col + width - 1) // segment_w

        rows_n = metadata.rows_number
        cols_n = metadata.columns_number
        segments_n_horizontally = math.ceil(cols_n / segment_w)

        file_path = os.path.join(self._data_dir, self._file_name)
        with open(file_path, "rb+", buffering=0) as f:
            fd = f.fileno()

            for segment_row in range(start_segment_row, end_segment_row + 1):
                for segment_col in range(start_segment_column, end_segment_column + 1):
                    start_row = max(segment_row * segment_h, row)
                    start_col = max(segment_col * segment_w, col)
                    end_row = min((segment_row + 1) * segment_h, row + height)
                    end_col = min((segment_col + 1) * segment_w, col + width)

                    local_y0 = start_row - segment_row * segment_h
                    local_x0 = start_col - segment_col * segment_w
                    local_h = end_row - start_row
                    local_w = end_col - start_col
                    if local_h <= 0 or local_w <= 0:
                        continue

                    # width of the stored segment (last column can be smaller)
                    seg_w_cur = segment_w
                    if segment_col == segments_n_horizontally - 1:
                        seg_w_cur = cols_n % segment_w or segment_w

                    seg_base = self._coords_to_file_position(segment_row, segment_col)

                    bytes_per_row = local_w * third * SINGLE_CELL_SIZE

                    # write rectangle row-by-row
                    for i in range(local_h):
                        seg_r = local_y0 + i
                        byte_off = seg_base + ((seg_r * seg_w_cur + local_x0) * third * SINGLE_CELL_SIZE)

                        src_row = (start_row - row) + i
                        src_col0 = start_col - col
                        src = fragment[src_row, src_col0 : src_col0 + local_w, :].astype(np.float32, copy=False)

                        written = os.pwrite(fd, src.tobytes(order="C"), byte_off)
                        if written != bytes_per_row:
                            raise IOError(f"Short write: wrote {written} bytes, expected {bytes_per_row}")

    def write_arbitrary_fragment(self, fragment: GridType, row: int, col: int) -> None:

        height, width = fragment.shape[0], fragment.shape[1]
        assert (
            row + height <= self._metadata.rows_number
        ), f"row + height cannot exceed number of rows of grid manager. Got row: {row}, height: {height}, while file's rows number is {self._metadata.rows_number}"
        assert (
            col + width <= self._metadata.columns_number
        ), f"col + width cannot exceed number of columns of grid manager. Got col: {col}, width: {width}, while file's colums number is {self._metadata.columns_number}"

        metadata = self._metadata
        segment_h, segment_w = metadata.segment_h, metadata.segment_w

        start_segment_row = row // segment_h
        start_segment_column = col // segment_w
        end_segment_row = (row + height - 1) // segment_h
        end_segment_column = (col + width - 1) // segment_w

        for segment_row in range(start_segment_row, end_segment_row + 1):
            for segment_col in range(start_segment_column, end_segment_column + 1):
                start_row = max(segment_row * segment_h, row)
                start_col = max(segment_col * segment_w, col)

                end_row = min((segment_row + 1) * segment_h, row + height)
                end_col = min((segment_col + 1) * segment_w, col + width)

                segment = self.read_segment(segment_row, segment_col)

                segment[
                    start_row - segment_row * segment_h : end_row - segment_row * segment_h,
                    start_col - segment_col * segment_w : end_col - segment_col * segment_w,
                ] = fragment[
                    start_row - row : end_row - row,
                    start_col - col : end_col - col,
                ]
                self.write_segment(
                    segment,
                    segment_row,
                    segment_col,
                )

    def _read_metadata(self) -> GridFileMetadata:
        """Read grid file metadata.
        Args:
            file_name (str): File name.
            data_dir (str): Folder with the file.
        Returns:
            metadata (GridFileMetadata): Metadata of the file.
        """
        DESIRED_METADATA_NUMBER = 10
        with open(os.path.join(self._data_dir, self._file_name), "r", encoding="latin1") as file:
            first_line = file.readline()[:-1]  # without \n
            metadata_bytes = file.tell()
            splitted = first_line.split(METADATA_SEPARATOR)

            assert (
                len(splitted) == DESIRED_METADATA_NUMBER
            ), f"Metadata line in the file is of wrong length {len(splitted)} instead of desired {DESIRED_METADATA_NUMBER}. Found metadata values: {splitted}"

            result = [
                int(splitted[0]),  # version
                int(splitted[1]),  # rows_number
                int(splitted[2]),  # cols_number
                float(splitted[3]),  # upper_left_longitude
                float(splitted[4]),  # upper_left_latitude
                float(splitted[5]),  # grid_density
                int(splitted[6]),  # segment_h
                int(splitted[7]),  # segment_w
                splitted[8],  # byteorder
                int(splitted[9]),  # third_dimension_size
            ]
            result += [metadata_bytes]

            return GridFileMetadata(*result)

    def _create_file(
        self,
        rows_number: int,
        columns_number: int,
        upper_left_longitude: float,
        upper_left_latitude: float,
        grid_density: float,
        segment_h: int,
        segment_w: int,
        third_dimension_size: int,
    ):
        """Creates a file for grid. If the file already exists, uses current one, if the metadata match.
         Args:
            rows_number (int): Number of rows of the whole gird (not just the segment).
            columns_number (int): Number of columns of the whole gird (not just the segment).
            upper_left_longitude (float): Longitude of the upper-left-most point.
            upper_left_latitude (float): Latitude of the upper-left-most point.
            grid_density (float): Distance between two distinct closest points in grid (in meters).
            segment_h (int): Height of segment (in grid rows).
            segment_w (int): Width of segment (in grid columns).
        Raises:
            FileExistsError: If the file already exists and has incompatible metadata parameters.
            FileNotFoundError: If values not provided and the file does not exist.
        """
        if os.path.exists(os.path.join(self._data_dir, self._file_name)):
            meta = self._read_metadata()

            if rows_number is not None and rows_number != meta.rows_number:
                raise FileExistsError(
                    f"The file already exists and has incompatible rows number: {rows_number} with given {meta.rows_number}"
                )
            if columns_number is not None and columns_number != meta.columns_number:
                raise FileExistsError(
                    f"The file already exists and has incompatible columns number: {columns_number} with given {meta.columns_number}"
                )
            return

        if None in [rows_number, columns_number, segment_h, segment_w]:
            raise FileNotFoundError(
                f"File was not found and not all arguments required for creating a new one were provided."
            )

        self._create_file_v1(
            rows_number,
            columns_number,
            upper_left_longitude,
            upper_left_latitude,
            grid_density,
            segment_h,
            segment_w,
            third_dimension_size,
        )

    def _create_file_v1(
        self,
        rows_number: int,
        columns_number: int,
        upper_left_longitude: float,
        upper_left_latitude: float,
        grid_density: float,
        segment_h: int,
        segment_w: int,
        third_dimension_size: int,
    ):
        """Create a file for points grid. File has the target size from the beginning (does not grow as a result of saving new segments)."""
        metadata_bytes = 0

        with open(os.path.join(self._data_dir, self._file_name), "w", encoding="latin1") as file:
            file.write(
                METADATA_SEPARATOR.join(
                    map(
                        str,
                        [
                            1,  # version
                            rows_number,
                            columns_number,
                            upper_left_longitude,
                            upper_left_latitude,
                            grid_density,
                            segment_h,
                            segment_w,
                            sys.byteorder,
                            third_dimension_size,
                        ],
                    )
                )
            )
            file.write("\n")
            metadata_bytes = file.tell()
            file.seek(metadata_bytes + rows_number * columns_number * third_dimension_size * SINGLE_CELL_SIZE - 1)
            file.write("\0")

    def _write_segment_v1(self, segment: GridType, segment_row: int, segment_col: int):

        self._assert_arguments_v1(segment_row, segment_col)

        if len(segment.shape) == 2:
            h, w = segment.shape
            new_segment = np.zeros((h, w, 3), dtype=np.float32)
            new_segment[:, :, 0] = segment
            segment = new_segment

        segments_n_vertically = math.ceil(self._metadata.rows_number / self._metadata.segment_h)
        segments_n_horizontally = math.ceil(self._metadata.columns_number / self._metadata.segment_w)

        given_segment_h, given_segment_w, _ = segment.shape

        rows_n = self._metadata.rows_number
        cols_n = self._metadata.columns_number

        if segment_row < segments_n_vertically - 1:
            assert (
                given_segment_h == self._metadata.segment_h
            ), f"Given segment has wrong height {given_segment_h} instead of desired {self._metadata.segment_h}."
        else:
            desired_height = rows_n % self._metadata.segment_h or self._metadata.segment_h
            assert (
                given_segment_h == desired_height
            ), f"Given segment has wrong height {given_segment_h}. It belongs to the last row, thus its height is expected to be {desired_height}."

        if segment_col < segments_n_horizontally - 1:
            assert (
                given_segment_w == self._metadata.segment_w
            ), f"Given segment has wrong width {given_segment_w} instead of desired {self._metadata.segment_w}."
        else:
            desired_width = cols_n % self._metadata.segment_w or self._metadata.segment_w
            assert (
                given_segment_w == desired_width
            ), f"Given segment has wrong width {given_segment_w}. It belongs to the last column, thus its width is expected to be {desired_width}."

        with open(os.path.join(self._data_dir, self._file_name), "rb+") as file:
            file.seek(self._coords_to_file_position(segment_row, segment_col))
            segment.astype(np.float32).tofile(file)

    def _read_segment_v1(self, segment_row: int, segment_col: int) -> GridType:

        self._assert_arguments_v1(segment_row, segment_col)

        rows_n = self._metadata.rows_number
        cols_n = self._metadata.columns_number

        segments_n_vertically = math.ceil(rows_n / self._metadata.segment_h)
        segments_n_horizontally = math.ceil(cols_n / self._metadata.segment_w)

        h = self._metadata.segment_h
        if segment_row == segments_n_vertically - 1:
            h = rows_n % self._metadata.segment_h or self._metadata.segment_h

        w = self._metadata.segment_w
        if segment_col == segments_n_horizontally - 1:
            w = cols_n % self._metadata.segment_w or self._metadata.segment_w

        SEEKED_SEGMENT_SHAPE = (h, w, self._metadata.third_dimension_size)

        with open(os.path.join(self._data_dir, self._file_name), "rb") as file:
            file.seek(self._coords_to_file_position(segment_row, segment_col))
            vector = np.fromfile(file, dtype=np.float32, count=np.prod(SEEKED_SEGMENT_SHAPE))
            return vector.reshape(SEEKED_SEGMENT_SHAPE)

    def _assert_arguments_v1(self, segment_row: int, segment_col: int):
        assert self._metadata.version == 1, "Given file does not support version 1."

        # verify segment's indices
        segments_n_vertically = math.ceil(self._metadata.rows_number / self._metadata.segment_h)
        segments_n_horizontally = math.ceil(self._metadata.columns_number / self._metadata.segment_w)

        assert (segment_row, segment_col) < (
            segments_n_vertically,
            segments_n_horizontally,
        ), f"Given segment is out of bound. Grid consists of {segments_n_vertically}x{segments_n_horizontally} segments, but given coordinates are ({segment_row}, {segment_col})."

    def _coords_to_file_position(self, segment_row: int, segment_column: int) -> int:
        """Compute the absolute byte offset in the file where a given segment starts.

        The returned value is measured from the beginning of the file (so it already
        accounts for the metadata header). This function validates the provided
        segment coordinates against the file metadata and raises AssertionError via
        _assert_arguments_v1 if the coordinates are out of range.

        Args:
            segment_row (int): 0-based index of the segment row.
            segment_column (int): 0-based index of the segment column.

        Returns:
            int: Byte offset from the start of the file where the requested segment begins.
        """
        self._assert_arguments_v1(segment_row, segment_column)

        rows_n = self._metadata.rows_number
        cols_n = self._metadata.columns_number
        seg_h = self._metadata.segment_h
        seg_w = self._metadata.segment_w
        third_dimension_size = self._metadata.third_dimension_size

        full_segment_bytes = seg_h * seg_w * third_dimension_size * SINGLE_CELL_SIZE

        full_segments_per_row = cols_n // seg_w
        remainder_cols = cols_n % seg_w

        remainder_segment_bytes = 0
        if remainder_cols > 0:
            remainder_segment_bytes = seg_h * remainder_cols * third_dimension_size * SINGLE_CELL_SIZE

        row_bytes = full_segment_bytes * full_segments_per_row + remainder_segment_bytes

        position = self._metadata.metadata_bytes + segment_row * row_bytes

        current_row_h = seg_h
        segments_vert = math.ceil(rows_n / seg_h)
        if segment_row == segments_vert - 1:
            current_row_h = rows_n % seg_h or seg_h  # obsługa 0

        position += segment_column * (current_row_h * seg_w * third_dimension_size * SINGLE_CELL_SIZE)

        return position

    def deep_copy(self) -> GridFileMetadata:
        path, extension = os.path.splitext(self._file_name)

        epoch_time = int(time.time())
        copied_file_name = path + "_copy_" + str(epoch_time) + extension

        shutil.copyfile(os.path.join(self._data_dir, self._file_name), os.path.join(self._data_dir, copied_file_name))

        copied = GridManager(
            copied_file_name,
            self._metadata.rows_number,
            self._metadata.columns_number,
            self._metadata.upper_left_longitude,
            self._metadata.upper_left_latitude,
            self._metadata.grid_density,
            self._metadata.segment_h,
            self._metadata.segment_w,
            self._data_dir,
            self._metadata.third_dimension_size,
        )

        return copied

    def delete(self):
        os.remove(os.path.join(self._data_dir, self._file_name))

    def get_metadata(self) -> GridFileMetadata:
        """Get object metadata.

        Returns
        -------
        GridFileMetadata
            Independent copy of file metadata object.
        """
        return copy.deepcopy(self._metadata)
