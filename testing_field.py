from scraper import DataLoader
import time
from shapely import Polygon

polygon = Polygon([[
        0.13629913330078128,
        47.94831606158684
      ],
      [
        0.2647018432617188,
        47.94831606158684
      ],
      [
        0.2647018432617188,
        48.034937455397866
      ],
      [
        0.13629913330078128,
        48.034937455397866
      ],
      [
        0.13629913330078128,
        47.94831606158684
      ]
        
    ])


data_loader = DataLoader(1.0)

time_start = time.time()
city = "Kraków, Polska"

grid_manager = data_loader.load_city_grid(polygon)
print(f"Grid saving time for {city} equals: {round((time.time() - time_start), 2)}s")