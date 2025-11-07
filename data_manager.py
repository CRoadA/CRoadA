import pickle
import os
import numpy as np
from typing import Any
from dataclasses import dataclass
from enum import Enum
import math
import sys

class GRID_INDICES(Enum):
    IS_STREET = 0
    ALTITUDE = 1

METADATA_SEPARATOR = ";"

Grid = np.ndarray[(Any, Any, Any), Any] # point on Grid
"""Type of numpy array with points grid (row, col, value_index). Value_index should be taken from GRID_INDICES."""

GridFileMetadata = tuple[int, int, int, int, int, str]
"""Metadata of grid file.
    tuple:
    - version (int): Format version idenifier. An appropriate version of read/write function must be used (or just a general-purpose one).
    - rows_number (int): Number of rows of the whole gird (not just the segment).
    - columns_number (int): Number of columns of the whole gird (not just the segment).
    - segment_h (int): Height of segment (in grid rows).
    - segment_w (int): Width of segment (in grid columns).
    - metadata_bytes (int): Length of the metadata line (in bytes).
    - byteorder ("little"|"big"): Big-endian or little-endian."""

def create_file(rows_number: int, columns_number: int, segment_h: int, segment_w: int, file_name: str, data_dir: str = "grids"):
    """Creates a file for grid.
    Args:
        rows_number (int): Number of rows of the whole gird (not just the segment).
        columns_number (int): Number of columns of the whole gird (not just the segment).
        segment_h (int): Height of segment (in grid rows).
        segment_w (int): Width of segment (in grid columns).
    Raises:
        FileExistsError: If the file already exists and has incompatible metadata parameters.
        """
    if os.path.exists(os.path.join(data_dir, file_name)):
        version, rows_n, cols_n, format_segment_h, format_segment_w, _, format_endian = _read_metadata(file_name, data_dir)
        
        assert rows_number == rows_n, f"The file already exists and has incompatible rows number: {rows_number} with given {rows_n}"
        assert columns_number == cols_n, f"The file already exists and has incompatible columns number: {columns_number} with given {cols_n}"
        assert segment_h == format_segment_h, f"The file already exists and has incompatible segment height: {segment_h} with given {format_segment_h}"
        assert segment_w == format_segment_w, f"The file already exists and has incompatible segment width: {segment_w} with given {format_segment_w}"
        # metadata matches -> nothing to create
        return
    
    _create_file_v1(rows_number, columns_number, segment_h, segment_w, file_name, data_dir)
    pass

def write_segment(segment: Grid, segment_row: int, segment_col: int, file_name: str, data_dir: str = "grids"):
    """Write segment of grid to a file. Requires preceeding creation of file with function create_file. General-purpose, file-format-agnostic function.
    Args:
        segment (Grid): Grid segment.
        segment_row (int): 0-based row index of the segment (in terms of segments, not the grids columns).
        segment_col (int): 0-based column index of the segment (in terms of segments, not the grids columns).
        file_name (str): File name.
        data_dir (str): Folder with the file.
    """
    metadata = _read_metadata(file_name, data_dir)
    version, _, _, _, _, _, _ = metadata

    if version == 1:
        _write_segment_v1(segment, segment_row, segment_col, file_name, data_dir=data_dir, metadata=metadata)
    else:
        raise ValueError(f"Unsupported file version {version}")
    
def read_segment(segment_row: int, segment_col: int, file_name: str, data_dir: str = "grids"):
    """Read segment from given file.
    Args:
        segment_row (int): 0-based row index of the segment (in terms of segments, not the grids columns).
        segment_col (int): 0-based column index of the segment (in terms of segments, not the grids columns)."""
    metadata = _read_metadata(file_name, data_dir)
    version, _, _, _, _, _, _ = metadata

    if version == 1:
        return _read_segment_v1(segment_row, segment_col, file_name, data_dir=data_dir, metadata=metadata)
    else:
        raise ValueError(f"Unsupported file version {version}")


def _create_file_v1(rows_number: int, columns_number: int, segment_h: int, segment_w: int, file_name: str, data_dir: str = "grids"):
    """Create a file for points grid. Grid has  from the beginning the target size (does not grow as a result of saving segments)."""
    metadata_bytes = 0
    CELL_SIZE = 8 # in bytes

    with open(os.path.join(data_dir, file_name), "w") as file:
        file.write(METADATA_SEPARATOR.join(map(str, [
            1, # version
            rows_number,
            columns_number,
            segment_h,
            segment_w,
            sys.byteorder
        ])))
        file.write("\n")
        print(f"written address: {metadata_bytes + rows_number * columns_number * CELL_SIZE - 1}")
        file.seek(metadata_bytes + rows_number * columns_number * CELL_SIZE - 1)
        file.write('\0')



def _write_segment_v1(segment: Grid, segment_row: int, segment_col: int, file_name: str, data_dir: str = "grids", metadata: GridFileMetadata = None):
    if metadata is None:
        metadata = _read_metadata(file_name, data_dir)

    _, rows_n, cols_n, format_segment_h, format_segment_w, metadata_bytes,_ = metadata

    _assert_arguments_v1(segment_row, segment_col, file_name, data_dir=data_dir, metadata=metadata)

    # verify shape of given segment
    segments_n_vertically = math.ceil(rows_n / format_segment_h)
    segments_n_horizontally = math.ceil(cols_n / format_segment_w)
    given_segment_h, given_segment_w, _ = segment.shape

    # for segment's height
    if rows_n % given_segment_h == 0 \
        or segment_row < segments_n_vertically - 1: # if the segment shape should follow the format (typical case)
        assert given_segment_h == format_segment_h, f"Given segment has wrong height {given_segment_h} instead of desired {format_segment_h}."
    else: # when that is the last row of a grid not being a multiplicity of format_segment_h
        # the last segment has to be smaller
        desired_height = rows_n - (segments_n_vertically - 1) * format_segment_h
        assert given_segment_h == desired_height, f"Given segment has wrong height {given_segment_h}. It belongs to the last row, thus its height is expected to be {desired_height}."
    
    # for segment's width
    if cols_n % given_segment_w == 0 \
        or segment_col < segments_n_horizontally - 1: # if the segment shape should follow the format (typical case)
        assert given_segment_w == format_segment_w, f"Given segment has wrong width {given_segment_w} instead of desired {format_segment_w}."
    else: # when that is the last column of a grid not being a multiplicity of format_segment_w
        # the last segment has to be smaller
        desired_width = cols_n - (segments_n_horizontally - 1) * format_segment_w
        assert given_segment_w == desired_width, f"Given segment has wrong width {given_segment_w}. It belongs to the last column, thus its width is expected to be {desired_width}."

    CELL_SIZE = 8 # in bytes
    SEGMENT_SIZE = format_segment_h * format_segment_w * CELL_SIZE
    REMAINDER_SEGMENT_SIZE = format_segment_h * cols_n % format_segment_w * CELL_SIZE
    ROW_SIZE = SEGMENT_SIZE + REMAINDER_SEGMENT_SIZE

    with open(os.path.join(data_dir, file_name), "rb+") as file:
        # skip the metadata line
        file.seek(metadata_bytes)

        base = file.tell()
        # advance to proper row
        file.seek(base + segment_row * ROW_SIZE + segment_col * SEGMENT_SIZE)
        segment.astype(np.float32).tofile(file)


def _read_segment_v1(segment_row: int, segment_col: int, file_name: str, data_dir: str = "grids", metadata: GridFileMetadata = None):
    if metadata is None:
        metadata = _read_metadata(file_name, data_dir)

    _, rows_n, cols_n, format_segment_h, format_segment_w, metadata_bytes, _ = metadata

    _assert_arguments_v1(segment_row, segment_col, file_name, data_dir=data_dir, metadata=metadata)

    segments_n_vertically = math.ceil(rows_n / format_segment_h)
    segments_n_horizontally = math.ceil(cols_n / format_segment_w)

    CELL_SIZE = 8 # in bytes
    SEGMENT_SIZE = format_segment_h * format_segment_w * CELL_SIZE
    REMAINDER_SEGMENT_SIZE = format_segment_h * cols_n % format_segment_w * CELL_SIZE
    ROW_SIZE = SEGMENT_SIZE + REMAINDER_SEGMENT_SIZE
    print(f"(segment_col, segment_row): ({segment_col}, {segment_row})")
    print(f"(cols_n, rows_n): ({cols_n}, {rows_n})")
    if segment_col == segments_n_horizontally - 1:
        if segment_row == segments_n_vertically - 1:
            SEEKED_SEGMENT_SHAPE = (rows_n % format_segment_h, cols_n % format_segment_w, 2)
        else:
            SEEKED_SEGMENT_SHAPE = (format_segment_h, cols_n % format_segment_w, 2)
    elif segment_row == segments_n_vertically - 1:
        SEEKED_SEGMENT_SHAPE = (rows_n % format_segment_h, format_segment_w, 2)
    else:
        SEEKED_SEGMENT_SHAPE = (format_segment_h, format_segment_w, 2)

    print(f"SEEKED_SEGMENT_SHAPE: {SEEKED_SEGMENT_SHAPE}")
    with open(os.path.join(data_dir, file_name), "rb") as file:
        base = metadata_bytes
        # advance to proper row
        file.seek(base + segment_row * ROW_SIZE + segment_col * SEGMENT_SIZE)
        vector = np.fromfile(file, dtype=np.float32, count=np.prod(SEEKED_SEGMENT_SHAPE))
        return vector.reshape(SEEKED_SEGMENT_SHAPE)


def _assert_arguments_v1(segment_row: int, segment_col: int, file_name: str, data_dir: str = "grids", metadata: GridFileMetadata = None):
    if metadata is None:
        metadata = _read_metadata(file_name, data_dir)

    version, rows_n, cols_n, format_segment_h, format_segment_w, metadata_bytes, format_endian = metadata
    assert version == 1, "Given file does not support version 1."

    # verify segment's indices
    segments_n_vertically = math.ceil(rows_n / format_segment_h)
    segments_n_horizontally = math.ceil(cols_n / format_segment_w)

    assert (segment_row, segment_col) < (segments_n_vertically, segments_n_horizontally), f"Given segment is out of bound. Grid consists of {segments_n_vertically}x{segments_n_horizontally} segments, but given coordinates are ({segment_row}, {segment_col})."


def _read_metadata(file_name: str, data_dir: str = "grids") -> GridFileMetadata:
    """Read grid file metadata.
    Args:
        file_name (str): File name.
        data_dir (str): Folder with the file.
    Returns:
        metadata (GridFileMetadata): Metadata of the file.
    """
    DESIRED_METADATA_NUMBER = 7
    with open(os.path.join(data_dir, file_name), "r") as file:
        first_line = file.readline()
        metadata_bytes = file.tell()
        splitted = first_line.split(METADATA_SEPARATOR)
        result = list(map(int, splitted[:-1] + [metadata_bytes]))
        result += [splitted[-1]]
        assert len(result) == DESIRED_METADATA_NUMBER, f"Metadata line in the file is of wrong length {len(result)} instead of desired {DESIRED_METADATA_NUMBER}. Found metadata values: {result}"
        return result