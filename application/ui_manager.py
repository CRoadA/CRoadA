from collections.abc import Callable
from ..graph_remaker.prediction_statistics import PredictionStatistics

class UIManager:

    def __init__(self, on_prediction_request: Callable[[dict], None]):
        self.on_prediction_request = on_prediction_request
        # TODO
        # 1. display map
        # 2. display place for result
        # 3. display "predict" button
        # 4. invoke on_prediction_request when button pressed
        

    def display_prediction(self, geo_json: dict, stats: PredictionStatistics):
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

    # there could be also some possibility to move to some city by typing it in.