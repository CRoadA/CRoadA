from grid_manager import GridManager
from trainer.model import InputGrid, OutputGrid, Model
from application.ui_manager import UIManager
from graph_remaker.data_analyser import DataAnalyser
from scraper.data_loader import DataLoader
from shapely import Polygon
import sys
from PyQt6.QtWidgets import QApplication
import qasync
import asyncio
from scraper.locator import Locator
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable

class Application(QApplication):

    _input_grid_manager: GridManager[InputGrid] | None
    _output_grid_manager: GridManager[OutputGrid] | None
    _model: Model
    _data_loader : DataLoader
    _data_analyser: DataAnalyser
    _ui_manager: UIManager


    def __init__(self):
        super().__init__(sys.argv)
        self._input_grid_manager = None
        self._output_grid_manager = None
        # self._model = Model() # throws, cause it's abstract class
        self._data_loader = DataLoader(grid_density=1.0)
        self._data_analyser = DataAnalyser()
        self._ui_manager = UIManager(self.handle_prediction_request, self.find_city_location)
    

    def show_window(self):
        self._ui_manager.window.show()
        

    async def handle_prediction_request(self, geo_json: dict):
        coordinates = geo_json.get("coordinates")[0]
        print(coordinates)
        area = Polygon(coordinates)
        loop = asyncio.get_running_loop()
        try:
            self._input_grid_manager = await loop.run_in_executor(
                None, 
                lambda : self._data_loader.load_city_grid(
                    area,
                    file_name=None,
                    on_progress=self._ui_manager.display_status
                )
            )
        except FileExistsError as e:
            print(f"Error while loading a city: {e.args[0]}, {type(e)}")
            message = f"{e.args[1]} has been marked recently."
            self._ui_manager.display_error(message)

        except ValueError as e:
            print(f"Error while loading a city: {e}, {type(e)}")
            self._ui_manager.display_error("Marked area is not a city.")

        except Exception as e:
            print(f"Error while loading a city: {e}, {type(e)}")
            self._ui_manager.display_error(str(e))
        # self._input_grid_manager = created_oneS
        # self._output_grid_manager = self._model.predict(self._input_grid_manager)
        # pred_geo_json, stats = self._data_analyser.get_GeoJSON_and_statistics(self._output_grid_manager)
        # self._ui_manager.display_prediction(pred_geo_json, stats)


    async def find_city_location(self, geo_json : dict):
        city_name = geo_json.get("cityName")
        if city_name is None:
            self._ui_manager.display_error("City name was not provided to the application")
            return None

        locator = Locator()
        try: 
            coords = locator.get_city_coords(city_name)
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