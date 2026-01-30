from grid_manager import GridManager
from trainer.model import InputGrid, OutputGrid, Model
from application.ui_manager import UIManager
from graph_remaker.data_analyser import DataAnalyser
from scraper.data_loader import DataLoader
from shapely import Polygon
import sys
import os
import glob
from trainer.clipping_model import ClippingModel, ClipModels
from PyQt6.QtWidgets import QApplication
import qasync
import asyncio
from scraper.locator import Locator
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable
from unidecode import unidecode 
import time

GRID_DENSITY = 1
GRID_FOLDER = "app_grids"

class Application(QApplication):

    _input_grid_manager: GridManager[InputGrid] | None
    _output_grid_manager: GridManager[OutputGrid] | None
    _model: Model
    _data_loader : DataLoader
    _data_analyser: DataAnalyser
    _ui_manager: UIManager
    _locator : Locator


    def __init__(self):
        super().__init__(sys.argv)
        self._input_grid_manager = None
        self._output_grid_manager = None
        self._model = ClippingModel(      # Define the model
            ClipModels.SHALLOWED_UNET,            # choose a model from ClipModels Enum (run the cell above to see possible models)
            clipping_size=256,          # size of the clipping (input to the model)
            clipping_surplus=64,        # surplus around the clipping (model gives an output smaller than input, so surplus is needed)
            path=os.path.join("models", "shallowed_unet_256_1m")     # where to save the model
        )
        self._data_loader = DataLoader(grid_density=GRID_DENSITY, data_dir=GRID_FOLDER)
        self._data_analyser = DataAnalyser()
        self._ui_manager = UIManager(self.save_marked_city, self.find_city_location, self.handle_prediction)
        self._locator = Locator()


    def show_window(self):
        self._ui_manager.window.show()


    async def save_marked_city(self, geo_json: dict):
        coordinates = geo_json.get("coordinates")[0]
        area = Polygon(coordinates)
        self._ui_manager.display_status(None, "Checking correctness of the marked area")

        city_name = self._locator.get_city_name(area)
        if city_name is None:
            self._ui_manager.display_error("Marked area is not a city.")
            return
            
        file_name = f"{unidecode(city_name).lower().replace(" ", "_")}.dat"
        loop = asyncio.get_running_loop()
        try:
            self._input_grid_manager = await loop.run_in_executor(
                None, 
                lambda : self._data_loader.load_city_grid(
                    area,
                    file_name=file_name,
                    on_progress=self._ui_manager.display_status
                )
            )
        except FileExistsError as e:
            print(f"Error while loading a city: {e}, {type(e)}")
            message = f"{city_name} has been marked recently."
            self._ui_manager.display_error(message)

        except ValueError as e:
            print(f"Error while loading a city: {e}, {type(e)}")
            self._ui_manager.display_error("Marked area is not a city.")

        except Exception as e:
            print(f"Error while loading a city: {e}, {type(e)}")
            self._ui_manager.display_error(str(e))

        self._ui_manager.window.prediction_button_activation(True)


    async def find_city_location(self, geo_json : dict):
        city_name = geo_json.get("cityName")
        if city_name is None:
            self._ui_manager.display_error("City name was not provided to the application")
            return None

        try: 
            coords = self._locator.get_city_coords(city_name)
            if coords is None:
                self._ui_manager.display_error(f"City: {city_name} does not exist")
                return None
            return coords
        except (GeocoderTimedOut, GeocoderUnavailable):
            print(f"Connection error: {e}")
            self._ui_manager.display_error(f"Connection to the geopy server is unavailable")
        
        except GeocoderServiceError as e:
            print(f"Server error: {e}")
            self._ui_manager.display_error(f"Geopy server error.")
            
        except Exception as e:
            print(f"Error: {e}")
            self._ui_manager.display_error(str(e))


    def handle_prediction(self):
        if self._input_grid_manager:
            self._ui_manager.display_status(None, "Predicting output for marked city...")
            time.sleep(2)
            self._output_grid_manager = self._model.predict(self._input_grid_manager)
            # pred_geo_json, stats = self._data_analyser.get_GeoJSON_and_statistics(self._output_grid_manager)
            # self._ui_manager.display_prediction(pred_geo_json, stats)
            self._ui_manager.display_status(None, "Koniec działania")
        else:
            self._ui_manager.display_error("Prediction is not enabled.")
        self.prediction_cleaner()


    def prediction_cleaner(self):
        self._input_grid_manager = None
        if os.path.exists(GRID_FOLDER):
            grids = glob.glob(os.path.join(GRID_FOLDER, "*.dat"))
            for file in grids:
                try:
                    os.remove(file)
                    print(f"Usunięto plik: {file}")
                except OSError as e:
                    print(f"Błąd przy usuwaniu {file}: {e}")


if __name__ == "__main__":
    try:
        app = Application()

        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)

        app.show_window()

        with loop:
            loop.run_forever()
            
    except Exception as e:
        print(f"Error starting application: {e}")