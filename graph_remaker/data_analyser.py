import networkx as nx
import json
from shapely.geometry import mapping

from grid_manager import GridManager
from trainer.model import PredictGrid
from .prediction_statistics import PredictionStatistics
from .memory_wise import LargeGridProcessor
from curve_analizer.curve_analizer import CurvatureAnalyzer


class DataAnalyser:

    def __init__(self):
        pass

    def get_GeoJSON_and_statistics(self, grid_manager: GridManager[PredictGrid]) -> tuple[dict, PredictionStatistics]:
        """Retrieve GeoJSON and statistics of generated streets from Models OutputGrid.

        Parameters
        ----------
        grid_manager : GridManager[PredictGrid]
            Output from model to process.

        Returns
        -------
        tuple[dict, PredictionStatistics]
            GeoJSON and statistics of given OutputGrid.

        """
        processor = LargeGridProcessor(grid_manager)
        G = processor.run()

        geojson = self._graph_to_geojson(G)

        stats = self._calculate_statistics(G)

        return geojson, stats

    def _graph_to_geojson(self, G: nx.MultiDiGraph) -> dict:
        """Converts NetworkX graph to GeoJSON FeatureCollection."""
        features = []

        for u, v, k, data in G.edges(keys=True, data=True):
            if 'geometry' in data:
                geom = mapping(data['geometry'])

                properties = {
                    "u": u,
                    "v": v,
                    "slope": data.get("slope", 0.0),
                    "width": data.get("width", 0.0)
                }

                features.append({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": properties
                })

        return {
            "type": "FeatureCollection",
            "features": features
        }

    def _calculate_statistics(self, G: nx.MultiDiGraph) -> PredictionStatistics:
        """Extracts physical properties and runs curvature analysis."""

        max_steepnesses = []
        for u, v, data in G.edges(data=True):
            if 'slope' in data:
                max_steepnesses.append(float(data['slope']))

        analyzer = CurvatureAnalyzer(G)
        curvature_data = analyzer.analyze_curvature(max_radius=1000.0)

        all_radii = curvature_data.get('street_curvature', []) + curvature_data.get('junction_turns', [])

        return PredictionStatistics(
            max_steepnesses=max_steepnesses,
            min_turning_radiuses=all_radii
        )