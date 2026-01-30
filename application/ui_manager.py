from collections.abc import Callable
from graph_remaker.prediction_statistics import PredictionStatistics
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
from application.MapWindow import MapWindow
import sys


from collections.abc import Callable
from application.main_window import MainWindow
from application.main_window import RESULTS_TYPE

class UIManager:

    def __init__(self, on_save_marked_city: Callable[[dict], None], on_city_location: Callable[[dict], None], on_handle_prediction: Callable[[None], None]):
        self.window = MainWindow(on_save_marked_city, on_city_location, on_handle_prediction)
        


    def display_prediction(self, geo_json: dict):#, stats: PredictionStatistics):
        """Shows obtained GeoJSON with predicted city mesh and statistics of the prediction. Ends indication of loading on client.

        Parameters
        ----------
        geo_json : dict
            GeoJson to display.
        stats : PredictionStatistics
            Statistics of the prediction.
        """
        # TODO
        pass


    def display_error(self, error_message : str):
        self.window.set_results(error_message, RESULTS_TYPE.ERROR)

    
    def display_status(self, saved_percentage : float | None, message : str):
        if saved_percentage is not None:
            self.window.set_results(f"{message} - {round(saved_percentage, 2)}%", RESULTS_TYPE.STATUS)
        else:
            self.window.set_results(message, RESULTS_TYPE.STATUS)
