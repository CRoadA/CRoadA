from grid_manager import GridManager
from trainer.model import InputGrid, OutputGrid, Model
from ui_manager import UIManager
from graph_remaker.data_analyser import DataAnalyser

class Application:

    _input_grid_manager: GridManager[InputGrid] | None
    _output_grid_manager: GridManager[OutputGrid] | None
    _model: Model
    _data_analyser: DataAnalyser
    _ui_manager: UIManager

    def __init__(self):
        self._input_grid_manager = None
        self._output_grid_manager = None
        # self._model = Model() # throws, cause it's abstract class
        self._data_analyser = DataAnalyser()
        self._ui_manager = UIManager(self.handle_prediction_request)

    def handle_prediction_request(self, geo_json: dict):
        # TODO make GridManager[InputGrid] from given GeoJSON
        # self._input_grid_manager = created_one
        # self._output_grid_manager = self._model.predict(self._input_grid_manager)
        # pred_geo_json, stats = self._data_analyser.get_GeoJSON_and_statistics(self._output_grid_manager)
        # self._ui_manager.display_prediction(pred_geo_json, stats)
        raise NotImplementedError()
        



Application()