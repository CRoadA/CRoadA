from grid_manager import GridManager
from trainer.model import InputGrid, PredictGrid, Model
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
import cv2 
import numpy as np
from trainer.model import PREDICT_GRID_INDICES
import math


GRID_DENSITY = 1
GRID_DIRECTORY = "app_grids"

class Application(QApplication):

    _input_grid_manager: GridManager[InputGrid] | None
    _output_grid_manager: GridManager[PredictGrid] | None
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
            ClipModels.UNET,            # choose a model from ClipModels Enum (run the cell above to see possible models)
            clipping_size=256,          # size of the clipping (input to the model)
            clipping_surplus=64,        # surplus around the clipping (model gives an output smaller than input, so surplus is needed)
            path=os.path.join("models", "unet_256_1m")     # where to save the model
        )
        self._data_loader = DataLoader(grid_density=GRID_DENSITY, data_dir=GRID_DIRECTORY)
        self._data_analyser = DataAnalyser()
        self._ui_manager = UIManager(self.save_marked_city, self.find_city_location, self.handle_prediction)
        self._locator = Locator()


    def show_window(self):
        self._ui_manager.window.show()

    
    def create_input_grid(self, area: Polygon, file_name: str):
        grid_manager = self._data_loader.load_city_grid(area, file_name=file_name, on_progress=self._ui_manager.display_status)

        self._data_loader.add_elevation_to_grid(grid_manager, on_progress=self._ui_manager.display_status)
        self._data_loader.add_residential_to_grid(grid_manager, area, on_progress=self._ui_manager.display_status)
        return grid_manager 


    async def save_marked_city(self, geo_json: dict):
        coordinates = geo_json.get("coordinates")[0]
        area = Polygon(coordinates)
        self._ui_manager.display_status(None, "Checking correctness of the marked area")

        try:
            city_name = self._locator.get_city_name(area)
        except Exception as e:
            print(f"Error while finding city name: {e}, {type(e)}")
            timestamp = str(int(time.time()))
            city_name = f"unknown_city_{timestamp}"

        if city_name is None:
            self._ui_manager.display_error("Marked area is not a city.")
            return
            
        file_name = unidecode(city_name).lower().replace(" ", "_") + ".grid_city"
        loop = asyncio.get_running_loop()
        try:
            self._input_grid_manager = await loop.run_in_executor(
                None, 
                lambda : self.create_input_grid(area, file_name)
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
            self._output_grid_manager = self._model.predict(self._input_grid_manager)
            try: 
                predicted_grid = self.get_predicted_grid()
                self._ui_manager.display_status(None, "Predicting has been finished")
                self._ui_manager.display_grid(predicted_grid) 
                # pred_geo_json, stats = self._data_analyser.get_GeoJSON_and_statistics(self._input_grid_manager)
                # self._ui_manager.display_prediction(pred_geo_json, stats)
            except Exception as e:
                message = "Error while getting predicted grid"
                print(f"{message}: {e}")
                self._ui_manager.display_error(message)
            finally:
                self.prediction_cleaner()
                self._output_grid_manager = None

            self._ui_manager.display_status(None, "Results were displayed")
        else:
            self._ui_manager.display_error("Prediction is not enabled.")
        self.prediction_cleaner()


    def prediction_cleaner(self):
        self._input_grid_manager = None
        if os.path.exists(GRID_DIRECTORY):
            grids = glob.glob(os.path.join(GRID_DIRECTORY, "*.grid_city"))
            for file in grids:
                try:
                    os.remove(file)
                    print(f"Usunięto plik: {file}")
                except OSError as e:
                    print(f"Błąd przy usuwaniu {file}: {e}")


    def get_predicted_grid(self) -> np.ndarray:
        if self._output_grid_manager is None:
            raise ValueError("Prediction should have been called first.")
        
        metadata = self._output_grid_manager.get_metadata()

        # assert (
        #     metadata.third_dimension_size == 3
        # ), "You have an outdated version of GridManager. Consider downloading grid files again to get all data..."

        fragment_row, fragment_col = math.ceil(metadata.rows_number * 0.2),  math.ceil(metadata.columns_number * 0.2)
        fragment_height, fragment_width = math.ceil(metadata.rows_number * 0.6), math.ceil(metadata.columns_number * 0.6)

        segment_h, segment_w = metadata.segment_h, metadata.segment_w
        being_predicted = GridManager(
            f"test_{time.time()}.city_grid",
            fragment_height,
            fragment_width,
            0,
            0,
            metadata.grid_density,
            segment_h,
            segment_w,
            data_dir="grids/evaluation",
            third_dimension_size=self._model.input_third_dimension,
        )
        # read_arbitrary_fragment does not take care of memory size - if there are some problems - just use smaller fragment
        # Add IS_PREDICTED
        tmp = np.ones((fragment_height, fragment_width, self._model.input_third_dimension))
        tmp[:, :, 1 : self._model.input_third_dimension] = self._output_grid_manager.read_arbitrary_fragment(
            fragment_row, fragment_col, fragment_height, fragment_width
        )[:, :, : self._model.input_third_dimension - 1]

        being_predicted.write_arbitrary_fragment(tmp, 0, 0)  # for instance some segment in the middle

        result = self._model.predict(being_predicted)
        img = result.read_arbitrary_fragment(
            0, 0, fragment_height - self._model.get_input_grid_surplus(), fragment_width - self._model.get_input_grid_surplus()
        )[:, :, PREDICT_GRID_INDICES.IS_STREET]

        print(f"DEBUG: img.max(): {img.max()}")

        struct_el = np.ones((3, 3))
        dilated = cv2.dilate(img, struct_el, iterations=3)
        return dilated


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