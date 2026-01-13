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
from scraper.data_loader import DataLoader

def main():
    print("=== START SPRAWDZANIA ===")
    DIR = "check_dir"
    FILE = "check.grid"

    # Czyścimy folder testowy
    if os.path.exists(DIR):
        shutil.rmtree(DIR)
    os.makedirs(DIR)

    print("1. Inicjalizacja GridManagera (wymuszamy 2 warstwy: ulice i wysokość)...")
    gm = GridManager(FILE, rows_number=100, columns_number=100,
                     grid_density=10, segment_h=50, segment_w=50,
                     data_dir=DIR,
                     upper_left_longitude=424000,
                     upper_left_latitude=5546000,
                     third_dimension_size=2)

    print("2. Zapisywanie początkowych danych (kanał 0 = ulice)...")
    for r in range(2):
        for c in range(2):
            # Tworzymy segment o wymiarach (50, 50, 2)
            data_segment = np.zeros((50, 50, 2), dtype=np.float32)
            data_segment[:, :, 0] = 1.0  # Ustawiamy ulice na 1.0
            gm.write_segment(data_segment, r, c)

    print("3. Uruchamianie add_elevation_to_grid (Mockowane SRTM = 123.0m)...")
    loader = DataLoader(grid_density=10, segment_h=50, segment_w=50, data_dir=DIR)
    loader.add_elevation_to_grid(gm)

    print("4. Odczyt i weryfikacja wyników...")
    seg = gm.read_segment(0, 0)

    val_street = seg[0, 0, 0]
    val_elev = seg[0, 0, 1]
    actual_shape = seg.shape

    print(f"   Kształt segmentu: {actual_shape}")
    print(f"   Wartość ulicy [0,0,0]: {val_street} (oczekiwane: 1.0)")
    print(f"   Wartość wysokości [0,0,1]: {val_elev} (oczekiwane: 123.0)")

    correct_shape = (actual_shape == (50, 50, 2))
    correct_street = (val_street == 1.0)
    correct_elev = (val_elev == 123.0)

    if correct_shape and correct_street and correct_elev:
        print("\n=== SUKCES! Wszystko działa poprawnie. ===")
    else:
        print("\n=== BŁĄD! Wyniki się nie zgadzają. ===")
        if not correct_shape: print(f"Błąd: Zły kształt tablicy! Jest {actual_shape}")
        if not correct_street: print(f"Błąd: Dane ulic zostały nadpisane lub są błędne! Jest {val_street}")
        if not correct_elev: print(f"Błąd: Wysokość nie została poprawnie zapisana! Jest {val_elev}")

if __name__ == "__main__":
    main()