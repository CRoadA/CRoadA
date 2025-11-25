from scraper import DataLoader
import time


data_loader = DataLoader(1.0)

time_start = time.time()
city = "Kraków, Polska"

grid_manager = data_loader.load_city_grid(city, "cracow.dat")
print(f"Grid saving time for {city} equals: {round((time.time() - time_start), 2)}s")