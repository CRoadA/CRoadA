from grid_manager import GridManager
from trainer.model import OutputGrid
from graph_remaker.prediction_statistics import PredictionStatistics

class DataAnalyser:

    def __init__(self):
        pass

    def get_GeoJSON_and_statistics(self, grid_manager: GridManager[OutputGrid]) -> tuple[dict, PredictionStatistics]:
        """Retrieve GeoJSON and statistics of generated streets from Models OutputGrid.

        Parameters
        ----------
        grid_manager : GridManager[OutputGrid]
            Output from model to process.

        Returns
        -------
        tuple[dict, PredictionStatistics]
            GeoJSON and statistics of given OutputGrid.

        """

        # TODO
        raise NotImplementedError()
