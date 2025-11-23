import os
import pickle
from scraper.geometry_processor import GeometryProcessor
import osmnx as ox
import geopandas as gpd
from scraper.rasterizer import Rasterizer
from scraper.graph_loader import GraphLoader
import matplotlib.pyplot as plt
from shapely import LineString


class GridBuilder():
    def __init__(self, folder="grids"):
        self.folder = folder
        os.makedirs(f"{self.folder}", exist_ok=True)
        self.loader = GraphLoader()
        self.geometry_processor = GeometryProcessor()
        self.rasterizer = Rasterizer()


    def load_pickle_grid(self, pickle_file_name):
        try:
            with open(f"{self.folder}/{pickle_file_name}", "rb") as pickle_file:
                grid = pickle.load(pickle_file)
            return grid
        except PermissionError as e:
            print(f"Permission denied while opening file: {pickle_file_name}", exc_info=True)
            return None
        except FileNotFoundError:
            print(f"File not found: {pickle_file_name}")
            return None
        except Exception as e:
            print(f"Unexpected error while loading {pickle_file_name}: {e}")
            return None


    def save_pickle_file(self, file_name, grid) -> bool:
        try:
            with open(f"{self.folder}/{file_name}", "wb") as pickle_file:
                pickle.dump(grid, pickle_file)
            return True
        except PermissionError as e:
            print(f"Permission denied while saving a file: {self.folder}, {str(e)}")
            return False
        except pickle.PickleError as e:
            print(f"Pickle serialization error: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error while saving a file: {e}")
            return False
        

    def get_city_grid(self, city_name):
        graph = self.loader.load_graph(city_name)

        for u, v, k, data in list(graph.edges(data=True, keys=True)):
            linestring = LineString([[graph.nodes[u]["x"], graph.nodes[u]["y"]], [graph.nodes[v]["x"], graph.nodes[v]["y"]]])
            data["geometry"] = linestring
        edges = self.loader.get_edges_measurements(graph)

        gdf_edges = self.loader.convert_to_gdf(edges)

        gdf_edges["geometry"] = gdf_edges.apply(lambda row: self.geometry_processor.get_edge_polygon(row), axis=1)

        grid = self.rasterizer.get_rasterize_roads(gdf_edges)

        return grid
    

    def get_city_roads(self, city_name):
        graph = self.loader.load_graph(city_name)

        for u, v, k, data in list(graph.edges(data=True, keys=True)):
            linestring = LineString([[graph.nodes[u]["x"], graph.nodes[u]["y"]], [graph.nodes[v]["x"], graph.nodes[v]["y"]]])
            data["geometry"] = linestring
        edges = self.loader.get_edges_measurements(graph)

        gdf_edges = self.loader.convert_to_gdf(edges)

        gdf_edges["geometry"] = gdf_edges.apply(lambda row: self.geometry_processor.get_edge_polygon(row), axis=1)

        return gdf_edges
    
    
    def show_grid(self, grid, city_name):
        plt.imshow(grid, cmap="gray")
        plt.title(f"Siatka: {city_name}")
        plt.show()


    def show_grid_fragment(self, grid, city_name, start_indexes : tuple, size : int):
        x = start_indexes[0]
        y = start_indexes[1]
        plt.imshow(grid[x : x + size, y : y + size], cmap="gray")
        plt.title(f"Siatka: {city_name}")
        plt.show()

