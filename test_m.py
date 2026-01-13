import numpy as np
from grid_manager import GRID_INDICES
from graph_remaker.morphological_remaker import discover_streets


def test_discovery_with_synthetic_data():
    grid = np.zeros((100, 100, 2), dtype=np.float32)
    grid[20:80, 45:55, 0] = 1.0
    for i in range(20, 80):
        grid[i, 45:55, 1] = (i - 20) * 0.1

    res = discover_streets(grid)
    streets = res[2] + res[3]

    print(f"Liczba wykrytych dróg: {len(streets)}")
    if streets:
        s = streets[0]
        print(f"--- Parametry drogi ---")
        print(f"Szerokość (oczekiwana ~10): {getattr(s, 'width', 'Brak pola')}")
        print(f"Nachylenie podłużne (oczekiwane ~0.1): {getattr(s, 'max_longitudinal_slope', 'Brak pola')}")
        print(f"Nachylenie poprzeczne (oczekiwane ~0.0): {getattr(s, 'max_transversal_slope', 'Brak pola')}")
        print(f"Liczba punktów w linestringu: {len(s.linestring)}")


if __name__ == "__main__":
    test_discovery_with_synthetic_data()