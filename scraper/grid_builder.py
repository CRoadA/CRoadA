import os
import pickle
from scraper.geometry_processor import GeometryProcessor
from scraper.rasterizer import Rasterizer
from scraper.graph_loader import GraphLoader
import matplotlib.pyplot as plt
from shapely import LineString, Polygon, Point
import numpy as np
import networkx as nx
from shapely.ops import linemerge
import math
import pandas as pd
import geopandas as gpd
import osmnx as ox

from typing import Union
from networkx import MultiDiGraph, MultiGraph
from geopy.geocoders import Nominatim


class GridBuilder():
    def __init__(self, folder="grids"):
        self.folder = folder
        os.makedirs(f"{self.folder}", exist_ok=True)
        self.loader = GraphLoader()
        self.geometry_processor = GeometryProcessor()
        self.rasterizer = Rasterizer()


    def load_pickle_grid(self, pickle_file_name : str) -> np.ndarray | None:
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


    def save_pickle_file(self, file_name : str, grid : np.ndarray) -> bool:
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
        

    def get_city_grid(self, area : Union[str, Polygon]):
        if isinstance(area, str):
            graph = self.loader.load_graph(area)
            new_crs = self.get_UTM_crs(graph)
        elif isinstance(area, Polygon):
            graph = self.loader.load_graph_from_polygon(area)
            new_crs = self.get_UTM_crs(area)
        else:
            raise TypeError("Argument 'area' must be a city name of type str or a Polygon")

        for u, v, k, data in list(graph.edges(data=True, keys=True)):
            linestring = LineString([[graph.nodes[u]["x"], graph.nodes[u]["y"]], [graph.nodes[v]["x"], graph.nodes[v]["y"]]])
            data["geometry"] = linestring
        edges = self.loader.get_edges_measurements(graph)

        gdf_edges = self.loader.convert_to_gdf(edges, new_crs)

        gdf_edges["geometry"] = gdf_edges.apply(lambda row: self.geometry_processor.get_edge_polygon(row), axis=1)

        grid = self.rasterizer.get_rasterize_roads(gdf_edges)

        return grid
    

    def get_city_roads(self, area : Union[str, Polygon], residential_max_radius : float = 30.0, shortest_node_distance : float = 10.0) -> gpd.GeoDataFrame:
        global_crs = None
        if isinstance(area, str):
            graph = self.loader.load_graph(area)
            # new_crs = self.get_UTM_crs(graph)
        elif isinstance(area, Polygon):
            graph = self.loader.load_graph_from_polygon(area)
            global_crs = self.get_UTM_crs(area)
        else:
            raise TypeError("Argument 'area' must be a city name of type str or a Polygon")

        for u, v, k, data in list(graph.edges(data=True, keys=True)):
            linestring = LineString([[graph.nodes[u]["x"], graph.nodes[u]["y"]], [graph.nodes[v]["x"], graph.nodes[v]["y"]]])
            data["geometry"] = linestring

        G_projected = ox.project_graph(graph)

        G_projected = self.add_radius_for_edges(G_projected, shortest_node_distance)
        G_projected = self.unify_radius_on_graph(G_projected)

        G_final = ox.project_graph(G_projected, to_crs="EPSG:4326")

        edges = self.loader.get_edges_measurements(G_final, residential_max_radius)

        gdf_edges = self.loader.convert_to_gdf(edges, new_crs=global_crs)


        gdf_edges["geometry"] = gdf_edges.apply(lambda row: self.geometry_processor.get_edge_polygon(row), axis=1)
        return gdf_edges


    def get_UTM_crs(self, area : Union[Polygon, MultiDiGraph, MultiGraph]) -> str:
        if isinstance(area, MultiDiGraph | MultiGraph):
            min_lon, min_lat, max_lon, max_lat  = area.bounds
            center_point = Point([max_lon - min_lon, max_lat - min_lat])
        elif isinstance(area, Polygon):
            center_point = area.centroid
        else:
            raise TypeError("Argument 'area' must be a city graph type MultiDiGraph, MultiGraph or a Polygon")
        lon, lat = center_point.x, center_point.y
        zone = math.ceil((lon + 180)/6)
        if lat >= 0:
            epsg = 32600 + zone
        else:
            epsg = 32700 + zone

        return epsg
    

    def get_city_name(polygon):
        center_point = polygon.centroid

        geolocator = Nominatim(user_agent="CRoadA 1.0")
        location = geolocator.reverse((center_point.y, center_point.x))

        print(location.address)
        print(location.raw["address"])
        return location
            
    
    def show_grid(self, grid : np.ndarray, city_name : str):
        plt.imshow(grid, cmap="gray")
        plt.title(f"Siatka: {city_name}")
        plt.show()


    def show_grid_fragment(self, grid : np.ndarray, city_name : str, start_indexes : tuple[int, int], size : int):
        x = start_indexes[0]
        y = start_indexes[1]
        plt.imshow(grid[x : x + size, y : y + size], cmap="gray")
        plt.title(f"Siatka: {city_name}")
        plt.show()    
    

    def _is_same_street(self, graph : nx.MultiDiGraph, u : int, v : int, target_name : str | None, mode : str) -> bool:

        if target_name is None:
            return False

        if mode == 'succ':
            if not graph.has_edge(u, v): return False
            edges_dict = graph[u][v]
        else:
            if not graph.has_edge(v, u): return False
            edges_dict = graph[v][u]

        match_found = False
        
        for k, data in edges_dict.items():
            raw_name = data.get("name")
            
            if not raw_name:
                continue

            if isinstance(raw_name, list):
                if target_name in raw_name:
                    match_found = True
                    break
            else:
                if target_name == str(raw_name):
                    match_found = True
                    break
        
        return match_found
    

    def _get_distant_point(self, graph : nx.MultiDiGraph, start_node : int, direction_node : int, nodes_coords : dict[int, tuple[float, float]], mode : str, street_name : str | None, target_dist : float = 10.0) -> tuple[float, float] | None:
        """
            Wędruje od start_node w stronę direction_node (i dalej), aż uzbiera target_dist lub trafi na skrzyżowanie.
            Zwraca współrzędne (x, y) znalezionego punktu.
        """
        current_node = direction_node
        center_node = start_node
        accumulated_dist = self.geometry_processor._get_distance(nodes_coords[center_node], nodes_coords[current_node])
        

        if accumulated_dist >= target_dist:
            return nodes_coords[current_node]

        while accumulated_dist < target_dist:
            # jeżeli true to znaczy że liczba krawędzi wychodzących z tego węzła jest większa niż 2, więc jest to skrzyżowanie
            if graph.degree(current_node) > 2:
                break 


            if mode == 'succ':
                neighbors = list(graph.successors(current_node))
            else:
                neighbors = list(graph.predecessors(current_node))

            valid_neighbors = [n for n in neighbors if n != center_node]

            # ślepy zaułek
            if not valid_neighbors:
                break 

            # przy simplify=False i degree=2 zostaje na pewno jeden sąsiad
            next_node = valid_neighbors[0]
            
            # jeśli następny node prowadzi na inną ulicę to zwracamy poprzedni 
            if not self._is_same_street(graph, current_node, next_node, street_name, mode):
                break

            dist = self.geometry_processor._get_distance(nodes_coords[current_node], nodes_coords[next_node])
            accumulated_dist += dist
            
            center_node = current_node
            current_node = next_node
        
        if accumulated_dist < target_dist:
            return None

        return nodes_coords[current_node]
    
    


    def add_radius_for_edges(self, graph : nx.MultiDiGraph, lookahead_dist : float) -> nx.MultiDiGraph:
        nodes_coords = {n: (data['x'], data['y']) for n, data in graph.nodes(data=True)}
        
        for u, v, k in graph.edges(keys=True):
            graph.edges[u, v, k]['turning_radius'] = np.inf

        for v in graph.nodes():
            predecessors = list(graph.predecessors(v))
            successors = list(graph.successors(v))
            
            if not predecessors or not successors:
                continue
            
            # punkt centralny
            p_center = nodes_coords[v]

            for u in predecessors:
                name_pred = None
                edges_dict = graph[u][v] 

                for k, data in edges_dict.items():
                    raw_name = data.get("name")
                    
                    if raw_name and not isinstance(raw_name, list):
                        name_pred = str(raw_name)
                        break

                if not self._is_same_street(graph, v, u, name_pred, mode="pred"):
                    continue

                p_in = self._get_distant_point(graph, v, u, nodes_coords, mode="pred", street_name=name_pred, target_dist=lookahead_dist)

                if not p_in:
                    continue

                valid_candidates = []

                for w in successors:
                    if u == w: continue

                    name_succ = None
                    edges_dict = graph[v][w] 

                    for k, data in edges_dict.items():
                        raw_name = data.get("name")
                        
                        if raw_name and not isinstance(raw_name, list):
                            name_succ = str(raw_name)
                            break

                    if name_pred != name_succ:
                         continue

                    if not self._is_same_street(graph, v, w, name_succ, mode="succ"):
                        continue

                    p_out = self._get_distant_point(graph, v, w, nodes_coords, mode="succ", street_name=name_succ, target_dist=lookahead_dist)
                    
                    if not p_out:
                        continue

                    deviation = self.geometry_processor._get_deviation_angle(p_in, p_center, p_out)
                    
                    # żeby wybrać najlepszą opcję i zniwelować przypadek brania succ i pred na rozwidleniu drogi 
                    valid_candidates.append({
                        'w': w,
                        'p_out': p_out,
                        'deviation': deviation
                    })
                    

                # filtracja rozwidleń
                if not valid_candidates:
                    continue
                
                best_candidate = valid_candidates[0]
                if len(valid_candidates) > 1:
                    best_candidate = min(valid_candidates, key=lambda x: x['deviation'])

                p_out = best_candidate['p_out']
                w_node = best_candidate['w'] # dla logów

                r = self.geometry_processor.get_circle_radius(p_in, p_center, p_out)
                
                if r is not None and not np.isnan(r) and graph.has_edge(u, v):
                    for k in graph[u][v]:
                        if not graph.edges[u, v, k].get("junction"):
                            current_val = graph.edges[u, v, k]['turning_radius']
                            if r < current_val:
                                graph.edges[u, v, k]['turning_radius'] = r
                                
        return graph
    

    def unify_radius_on_graph(self, graph : nx.MultiDiGraph, attribute_name : str = "turning_radius") -> nx.MultiDiGraph:
        """
            Dla każdej nazwanej ulicy w grafie znajduje najmniejszy promień skrętu
            wśród jej segmentów i przypisuje go do wszystkich krawędzi tej ulicy.
        """
        
        edge_data = []
        for u, v, k, data in graph.edges(keys=True, data=True):
            raw_name = data.get("name", None)
            radius = data.get(attribute_name, np.inf)
            id = data.get("id", None)
            edge_data.append({"id" : id, 'u': u, 'v': v, 'k': k, 'name': raw_name, 'radius': radius})

        df = pd.DataFrame(edge_data)

        def get_group_key(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            if isinstance(val, list):
                return str(val[0])
            return str(val)

        df['group_key'] = df['name'].apply(get_group_key)
        valid_names_mask = df['group_key'].notna()
        
        # tworzymy słownik: { 'Marszałkowska': 2.5, 'Polna': 5.0, ... }
        min_radius_map = df[valid_names_mask].groupby('group_key')['radius'].min().to_dict()

        updated_count = 0
        
        for u, v, k, data in graph.edges(keys=True, data=True):
            raw_name = data.get("name", None)
            group_key = get_group_key(raw_name)

            # jeśli ulica ma nazwę i mamy dla niej wyliczone minimum
            if group_key in min_radius_map:
                min_val = min_radius_map[group_key]
                
                if min_val != np.inf:
                    graph.edges[u, v, k][attribute_name] = min_val
                    updated_count += 1

        print(f"Zaktualizowano '{attribute_name}' dla {updated_count} krawędzi.")
        return graph
    
