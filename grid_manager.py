import pickle
import os
import numpy as np
from typing import Any, Literal, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
import math
import sys
import copy


class GRID_INDICES:
    IS_STREET = 0
    ALTITUDE = 1


METADATA_SEPARATOR = ";"

Grid = np.ndarray[(Any, Any, 2), Any]  # point on Grid (row, col, value_index)
"""Type of numpy array with points grid (row, col, value_index). Value_index should be taken from GRID_INDICES."""


@dataclass
class GridFileMetadata():
    """Metadata of grid file.
    Attributes:
        version (int): Format version idenifier. An appropriate version of read/write function must be used (or just a general-purpose one).
        rows_number (int): Number of rows of the whole gird (not just the segment).
        columns_number (int): Number of columns of the whole gird (not just the segment).
        upper_left_longitude (float): Longitude of upper-left-most point in the file.
        upper_left_latitude (float): Latitude of upper-left-most point in the file.
        grid_density (float): Distance between two distinct closest points in grid (in meters).
        segment_h (int): Height of segment (in grid rows).
        segment_w (int): Width of segment (in grid columns).
        byteorder ("little"|"big"): Big-endian or little-endian.
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
    metadata_bytes: int


GridType = TypeVar('GridType', bound=np.ndarray)


class GridManager(Generic[GridType]):
    _file_name: str
    _data_dir: str

    _metadata: GridFileMetadata

    def __init__(self, file_name: str, rows_number: int | None = None, columns_number: int | None = None,
                 upper_left_longitude: float | None = None, upper_left_latitude: float | None = None,
                 grid_density: float | None = 1, segment_h: int = 5000, segment_w: int = 5000, data_dir: str = "grids"):
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
            data_dir (str): Folder with the file.
            """
        self._file_name = file_name
        self._data_dir = data_dir

        if not os.path.exists(self._data_dir):
            os.makedirs(self._data_dir)

        self._create_file(rows_number, columns_number, upper_left_longitude, upper_left_latitude, grid_density,
                          segment_h, segment_w)
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

    def _read_metadata(self) -> GridFileMetadata:
        """Read grid file metadata.
        Args:
            file_name (str): File name.
            data_dir (str): Folder with the file.
        Returns:
            metadata (GridFileMetadata): Metadata of the file.
        """
        DESIRED_METADATA_NUMBER = 9
        with open(os.path.join(self._data_dir, self._file_name), "r", encoding="utf-8") as file:
            first_line = file.readline()[:-1]  # without \n
            metadata_bytes = file.tell()
            splitted = first_line.split(METADATA_SEPARATOR)

            assert len(
                splitted) == DESIRED_METADATA_NUMBER, f"Metadata line in the file is of wrong length {len(splitted)} instead of desired {DESIRED_METADATA_NUMBER}. Found metadata values: {splitted}"

            result = [
                int(splitted[0]),  # version
                int(splitted[1]),  # rows_number
                int(splitted[2]),  # cols_number
                float(splitted[3]),  # upper_left_longitude
                float(splitted[4]),  # upper_left_latitude
                float(splitted[5]),  # grid_density
                int(splitted[6]),  # segment_h
                int(splitted[7]),  # segment_w
                splitted[8]  # byteorder
            ]
            result += [metadata_bytes]

            return GridFileMetadata(*result)

    def _create_file(self, rows_number: int, columns_number: int, upper_left_longitude: float,
                     upper_left_latitude: float, grid_density: float, segment_h: int, segment_w: int):
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
                    f"The file already exists and has incompatible rows number: {rows_number} with given {meta.rows_number}")
            if columns_number is not None and columns_number != meta.columns_number:
                raise FileExistsError(
                    f"The file already exists and has incompatible columns number: {columns_number} with given {meta.columns_number}")
            return

        if None in [rows_number, columns_number, segment_h, segment_w]:
            raise FileNotFoundError(
                f"File was not found and not all arguments required for creating a new one were provided.")

        self._create_file_v1(rows_number, columns_number, upper_left_longitude, upper_left_latitude, grid_density,
                             segment_h, segment_w)

    def _create_file_v1(self, rows_number: int, columns_number: int, upper_left_longitude: float,
                        upper_left_latitude: float, grid_density: float, segment_h: int, segment_w: int):
        """Create a file for points grid. File has the target size from the beginning (does not grow as a result of saving new segments)."""
        metadata_bytes = 0
        CELL_SIZE = 8  # ZMIANA: 8 bajtów (2 kanały * 4 bajty float32)

        with open(os.path.join(self._data_dir, self._file_name), "w", encoding="utf-8") as file:
            file.write(METADATA_SEPARATOR.join(map(str, [
                1,  # version
                rows_number,
                columns_number,
                upper_left_longitude,
                upper_left_latitude,
                grid_density,
                segment_h,
                segment_w,
                sys.byteorder
            ])))
            file.write("\n")
            metadata_bytes = file.tell()
            file.seek(metadata_bytes + rows_number * columns_number * CELL_SIZE - 1)
            file.write('\0')

    def _write_segment_v1(self, segment: GridType, segment_row: int, segment_col: int):

        self._assert_arguments_v1(segment_row, segment_col)

        if len(segment.shape) == 2:
            h, w = segment.shape
            new_segment = np.zeros((h, w, 2), dtype=np.float32)
            new_segment[:, :, 0] = segment
            segment = new_segment

        segments_n_vertically = math.ceil(self._metadata.rows_number / self._metadata.segment_h)
        segments_n_horizontally = math.ceil(self._metadata.columns_number / self._metadata.segment_w)

        given_segment_h, given_segment_w, _ = segment.shape

        rows_n = self._metadata.rows_number
        cols_n = self._metadata.columns_number

        if segment_row < segments_n_vertically - 1:
            assert given_segment_h == self._metadata.segment_h, f"Given segment has wrong height {given_segment_h} instead of desired {self._metadata.segment_h}."
        else:
            desired_height = rows_n % self._metadata.segment_h or self._metadata.segment_h
            assert given_segment_h == desired_height, f"Given segment has wrong height {given_segment_h}. It belongs to the last row, thus its height is expected to be {desired_height}."

        if segment_col < segments_n_horizontally - 1:
            assert given_segment_w == self._metadata.segment_w, f"Given segment has wrong width {given_segment_w} instead of desired {self._metadata.segment_w}."
        else:
            desired_width = cols_n % self._metadata.segment_w or self._metadata.segment_w
            assert given_segment_w == desired_width, f"Given segment has wrong width {given_segment_w}. It belongs to the last column, thus its width is expected to be {desired_width}."

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

        SEEKED_SEGMENT_SHAPE = (h, w, 2)

        with open(os.path.join(self._data_dir, self._file_name), "rb") as file:
            file.seek(self._coords_to_file_position(segment_row, segment_col))
            vector = np.fromfile(file, dtype=np.float32, count=np.prod(SEEKED_SEGMENT_SHAPE))
            return vector.reshape(SEEKED_SEGMENT_SHAPE)

    def _assert_arguments_v1(self, segment_row: int, segment_col: int):
        assert self._metadata.version == 1, "Given file does not support version 1."

        # verify segment's indices
        segments_n_vertically = math.ceil(self._metadata.rows_number / self._metadata.segment_h)
        segments_n_horizontally = math.ceil(self._metadata.columns_number / self._metadata.segment_w)

        assert (segment_row, segment_col) < (segments_n_vertically,
                                             segments_n_horizontally), f"Given segment is out of bound. Grid consists of {segments_n_vertically}x{segments_n_horizontally} segments, but given coordinates are ({segment_row}, {segment_col})."

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

        CELL_SIZE = 8  # bytes per stored value (2 * float32)

        full_segment_bytes = seg_h * seg_w * CELL_SIZE

        full_segments_per_row = cols_n // seg_w
        remainder_cols = cols_n % seg_w

        remainder_segment_bytes = 0
        if remainder_cols > 0:
            remainder_segment_bytes = seg_h * remainder_cols * CELL_SIZE

        row_bytes = full_segment_bytes * full_segments_per_row + remainder_segment_bytes

        position = self._metadata.metadata_bytes + segment_row * row_bytes

        current_row_h = seg_h
        segments_vert = math.ceil(rows_n / seg_h)
        if segment_row == segments_vert - 1:
            current_row_h = rows_n % seg_h or seg_h  # obsługa 0

        position += segment_column * (current_row_h * seg_w * CELL_SIZE)

        return position

    def get_metadata(self) -> GridFileMetadata:
        """Get object metadata.

        Returns
        -------
        GridFileMetadata
            Independent copy of file metadata object.
        """
        return copy.deepcopy(self._metadata)