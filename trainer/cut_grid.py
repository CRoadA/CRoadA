import math
import os
import numpy as np

from grid_manager import GridManager, GridType
from trainer.model import PredictGrid

def cut_from_grid_segments(
    grid_manager: GridManager, cut_start_x: int, cut_start_y: int, cut_size: tuple[int, int], surplus: int = 0, clipping: bool = False
) -> np.ndarray:
    """Create a cut grid from given grid manager by reading segments. No need to worry about cut_size exceeding boundaries - it is handled inside.

    Parameters
    ----------
    grid_manager : GridManager
        Grid manager to read segments from.
    cut_start_x : int
        Starting x coordinate of the cut.
    cut_start_y : int
        Starting y coordinate of the cut.
    cut_size : tuple[int, int]
        Size of the cut (rows, columns).
    surplus : int, optional
        Surplus size to consider around the cut, by default 0
    clipping : bool, optional
        Whether the cut is for clipping purposes, by default False
    Returns
    -------
    np.ndarray
        Cut grid as a numpy array.
    """
    # print(f"Requested cut at ({cut_start_x}, {cut_start_y}) of size {cut_size} with surplus {surplus}, clipping={clipping}")  # Debug print
    # Adjust surplus if it exceeds boundaries # TODO - Add padding if needed instead of reducing surplus
    surplus = max(min(surplus, cut_start_x, cut_start_y, cut_size[1], cut_size[0]), 0)

    # Load metadata
    metadata = grid_manager.get_metadata()

    if clipping:
        # Calculate start point with surplus
        cut_start_x = int(cut_start_x * (cut_size[1] - surplus) - (surplus / 2))
        cut_start_y = int(cut_start_y * (cut_size[0] - surplus) - (surplus / 2))
    else:
        cut_start_x = int(cut_start_x - surplus/2)
        cut_start_y = int(cut_start_y - surplus/2)
    cut_start_x = min(cut_start_x, metadata.columns_number - cut_size[1])
    cut_start_y = min(cut_start_y, metadata.rows_number - cut_size[0])
    
    # Calculate end point
    cut_end_x = min(cut_start_x + cut_size[0], metadata.columns_number)
    cut_end_y = min(cut_start_y + cut_size[1], metadata.rows_number)

    # Determine which segments to read
    segment_w, segment_h = metadata.segment_w, metadata.segment_h
    
    # print(f"Cut from segments: start_x={cut_start_x}, start_y={cut_start_y}, end_x={cut_end_x}, end_y={cut_end_y}")  # Debug print
    # print(f"Segment size: w={segment_w}, h={segment_h}")  # Debug print
    which_segment_start_x = int(cut_start_x // segment_w)
    which_segment_start_y = int(cut_start_y // segment_h)
    which_segment_end_x = int((cut_end_x - 1) // segment_w)
    which_segment_end_y = int((cut_end_y - 1) // segment_h)

    # print(f"Segments to read: start_x={which_segment_start_x}, start_y={which_segment_start_y}, end_x={which_segment_end_x}, end_y={which_segment_end_y}")  # Debug print
    cut_x = np.array([])  # Grid()
    cut_y = np.array([])
    # Go by segments and merge them into one bigger cut - first vertically, then horizontally.
    for indx_x in range(which_segment_start_x, which_segment_end_x + 1):
        for indx_y in range(which_segment_start_y, which_segment_end_y + 1):
            # memory issue -> adjust the maximal segment size or something -> we will have proper segment sizes anyway
            segment_y = grid_manager.read_segment(indx_y, indx_x)
            # print(f"Reading segment at ({indx_x}, {indx_y}) with shape {segment_y.shape}")  # Debug print

            # Merge segment_y into cut_y vertically.
            if indx_y == which_segment_start_y and indx_y == which_segment_end_y:
                cut_y = segment_y[cut_start_y % segment_h : cut_end_y % segment_h, :]
            elif indx_y == which_segment_start_y:
                cut_y = segment_y[cut_start_y % segment_h :, :]
            elif indx_y == which_segment_end_y:
                cut_y = np.vstack((cut_y, segment_y[: cut_end_y % segment_h, :]))
            else:
                cut_y = np.vstack((cut_y, segment_y))

        # Merge cut_y into cut_x horizontally.
        if indx_x == which_segment_start_x and indx_x == which_segment_end_x:
            cut_x = cut_y[:, cut_start_x % segment_w : cut_end_x % segment_w]
        elif indx_x == which_segment_start_x:
            cut_x = cut_y[:, cut_start_x % segment_w :]
        elif indx_x == which_segment_end_x:
            cut_x = np.hstack((cut_x, cut_y[:, : cut_end_x % segment_w]))
        else:
            cut_x = np.hstack((cut_x, cut_y))

    return cut_x

def write_clipping_to_grid_manager(
        grid_manager: GridManager[PredictGrid],
        clipping: np.ndarray,
        clipping_start_x: int,
        clipping_start_y: int
        ) -> None:
    
    metadata = grid_manager.get_metadata()
    segment_w, segment_h = metadata.segment_w, metadata.segment_h

    clipping_h, clipping_w = clipping.shape

    start_segment_row = clipping_start_y // segment_h
    start_segment_column = clipping_start_x // segment_w

    end_segment_row = clipping_start_y + clipping_h // segment_h
    end_segment_column = clipping_start_x + clipping_w // segment_w

    for segment_row in range(start_segment_row, end_segment_row + 1):
        for segment_col in range(start_segment_column, end_segment_column + 1):
            start_row = max(
                segment_row * segment_h,
                clipping_start_y
            )
            start_col = max(
                segment_col * segment_w,
                clipping_start_x
            )

            end_row = min(
                (segment_row + 1) * segment_h,
                clipping_start_y + clipping_h
            )
            end_col = min(
                (segment_col + 1) * segment_w,
                clipping_start_x + clipping_w
            )
            grid_manager.write_segment(
                clipping[
                    start_row - clipping_start_y : end_row - clipping_start_y,
                    start_col - clipping_start_x : end_col - clipping_start_x,
                ],
                segment_row,
                segment_col,
            )
    

def write_cut_to_grid_segments(
    cut: np.ndarray,
    cut_size: tuple[int, int],
    segment_w: int,
    segment_h: int,
    grid_density: int,
    cut_start_x: int,
    cut_start_y: int,
    from_file_path: str,
    to_directory: str,
) -> GridManager:
    """
    Write the cut grid into a new GridManager by segments.
    
    :param cut: The cut grid to write.
    :type cut: np.ndarray
    :param cut_size: Size of the cut (rows, columns).
    :type cut_size: tuple[int, int]
    :param segment_w: Segment width.
    :type segment_w: int
    :param segment_h: Segment height.
    :type segment_h: int
    :param cut_start_x: Starting x-coordinate of the cut.
    :type cut_start_x: int
    :param cut_start_y: Starting y-coordinate of the cut.
    :type cut_start_y: int
    :param from_file_path: File path where the cut grid comes from.
    :type from_file_path: str
    :return: A new GridManager instance containing the cut grid.
    :rtype: GridManager
    """
    cut_segment_rows = math.ceil(cut_size[0] / segment_h)  # number of segments in cut vertically
    cut_segment_columns = math.ceil(cut_size[1] / segment_w)  # number of segments in cut horizontally

    # Check if the cut grid file already exists
    file_name = f"{from_file_path}_cut_{cut_start_y}_{cut_start_x}_{cut_size[0]}_{cut_size[1]}.dat"
    file_path = os.path.join(to_directory, file_name)
    cut_grid: GridManager = None
    
    if os.path.isfile(file_path):
        cut_grid = GridManager(file_name, data_dir=to_directory)  # load grid manager
    else:
        cut_grid = GridManager(
            file_name,
            cut_size[0],
            cut_size[1],
            0,
            0,
            grid_density,
            segment_h,
            segment_w,
            to_directory,
        )
        for segment_row in range(cut_segment_rows):
            for segment_col in range(cut_segment_columns):
                cut_grid.write_segment(
                    cut[
                        segment_row * segment_h : min((segment_row + 1) * segment_h, cut_size[0]),
                        segment_col * segment_w : min((segment_col + 1) * segment_w, cut_size[1]),
                    ],
                    segment_row,
                    segment_col,
                )
    return cut_grid



def cut_from_cut(
    cut_input: np.ndarray, cut_start_x: int, cut_start_y: int, cut_size: tuple[int, int], surplus: int = 0, clipping: bool = False
) -> np.ndarray:
    """Create a cut grid from given cut grid as a numpy array. No need to worry about cut_size exceeding boundaries - it is handled inside.

    Parameters
    ----------
    cut_input : np.ndarray
        Input cut grid as a numpy array.
    cut_start_x : int
        Starting x coordinate of the cut.
    cut_start_y : int
        Starting y coordinate of the cut.
    cut_size : tuple[int, int]
        Size of the cut (rows, columns).
    surplus : int, optional
        Surplus size to consider around the cut, by default 0
    clipping : bool, optional
        Whether the cut is for clipping purposes, by default False
    Returns
    -------
    np.ndarray
        Cut grid as a numpy array.
    """
    # print(f"Requested cut at ({cut_start_x}, {cut_start_y}) of size {cut_size} with surplus {surplus}, clipping={clipping}")  # Debug print
    # Adjust surplus if it exceeds boundaries # TODO - Add padding if needed instead of reducing surplus
    surplus = max(min(surplus, cut_start_x, cut_start_y, cut_size[0], cut_size[1]), 0)

    if clipping:
        # Calculate start point with surplus
        cut_start_x = int(cut_start_x * (cut_size[1] - surplus) - (surplus / 2))
        cut_start_y = int(cut_start_y * (cut_size[0] - surplus) - (surplus / 2))
    else:
        cut_start_x = int(cut_start_x - surplus/2)
        cut_start_y = int(cut_start_y - surplus/2)
    cut_start_x = min(cut_start_x, cut_input.shape[1] - cut_size[1])
    cut_start_y = min(cut_start_y, cut_input.shape[0] - cut_size[0])
    
    # Calculate end point
    cut_end_x = min(cut_start_x + cut_size[1], cut_input.shape[1])
    cut_end_y = min(cut_start_y + cut_size[0], cut_input.shape[0])

    # Cut the cut
    cut_output = cut_input[cut_start_y : cut_end_y, cut_start_x : cut_end_x]

    return cut_output