import sys
import numpy as np
import os
import shutil
from unittest.mock import MagicMock

mock_srtm = MagicMock()
mock_geo_data = MagicMock()
mock_geo_data.get_elevation.return_value = 123.0
mock_srtm.get_data.return_value = mock_geo_data
sys.modules['srtm'] = mock_srtm

sys.path.append(os.getcwd())

from grid_manager import GridManager

from scraper import DataLoader


def main():
    print("=== START SPRAWDZANIA ===")
    DIR = "check_dir"
    FILE = "check.grid"

    if os.path.exists(DIR):
        shutil.rmtree(DIR)
    os.makedirs(DIR)

    print("1. Tworzenie pustego pliku grid...")
    gm = GridManager(FILE, rows_number=100, columns_number=100,
                     grid_density=10, segment_h=50, segment_w=50,
                     data_dir=DIR, upper_left_longitude=424000, upper_left_latitude=5546000)

    print("2. Zapisywanie danych 2D (ulice=1.0)...")
    for r in range(2):
        for c in range(2):
            data_2d = np.ones((50, 50))
            gm.write_segment(data_2d, r, c)
    print("3. Uruchamianie add_elevation_to_grid (mocked SRTM=123.0m)...")
    loader = DataLoader(grid_density=10, segment_h=50, segment_w=50, data_dir=DIR)
    loader.add_elevation_to_grid(gm)

    print("4. Odczyt i weryfikacja...")
    seg = gm.read_segment(0, 0)

    val_street = seg[0, 0, 0]
    val_elev = seg[0, 0, 1]

    print(f"   Wartość ulicy [0,0,0]: {val_street}")
    print(f"   Wartość wysokości [0,0,1]: {val_elev}")

    if seg.shape == (50, 50, 2) and val_street == 1.0 and val_elev == 123.0:
        print("\n=== SUKCES! Wszystko działa poprawnie. ===")
    else:
        print("\n=== BŁĄD! Wyniki się nie zgadzają. ===")


if __name__ == "__main__":
    main()