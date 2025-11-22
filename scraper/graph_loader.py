import numpy as np
import osmnx as ox
from shapely import LineString
import geopandas as gpd

class GraphLoader():
    def __init__(self, poland_crs = 2180, basic_crs = 4326):
        self.poland_crs = poland_crs
        self.basic_crs = basic_crs


    def load_graph(self, city_name):
        graph = ox.graph.graph_from_place(city_name, network_type="drive")
        return graph
    

    def get_all_highways(self, graph):
        highways = []
        for u, v, k, data in graph.edges(keys=True, data=True):
            if "highway" in data and data["highway"] not in highways and not isinstance(data["highway"], list):
                highways.append(data["highway"])
            elif isinstance(data["highway"], list):
                highways.extend(h for h in data["highway"] if h not in highways)

        return highways

    def get_avg_width(self, graph, highway):
        count_street = 0
        sum_width = 0

        for u, v, data in list(graph.edges(data=True)):
            width = 0
            if "highway" in data and ((not isinstance(data["highway"], list) and data["highway"] == highway) or (isinstance(data["highway"], list) and highway in data["highway"])):
                if "width" in data:
                    if isinstance(data["width"], list):
                        width = np.mean([float(x) for x in data["width"]])         
                    else:
                        width = float(data["width"])
                elif 'lanes' in data:
                    if isinstance(data["lanes"], list):
                        width = np.mean([int(x) for x in data["lanes"]]) * 3
                    else:
                        width = int(data['lanes']) * 3  # przyjmujemy 3 m na pas
                sum_width += width
                count_street += 1
                
        return sum_width/count_street

    def get_highways_width(self, graph):
        highways = self.get_all_highways(graph)
        avg_width = {}

        for highway in highways:
            avg_width[highway] = self.get_avg_width(graph, highway)
        
        return avg_width
    

    def get_edges_measurements(self, graph):

        # mapping - typ drogi : szerokość
        highway_width = self.get_highways_width(graph)

        edges_info = []
        lanes_width_counter = 0
        highway_width_counter = 0
        for u, v, k, data in graph.edges(keys=True, data=True):
            if 'length' in data:
                length = data['length']
            elif 'geometry' in data:
                if isinstance(data['geometry'], LineString):
                    # Używamy funkcji OSMnx do obliczenia długości geograficznej w metrach
                    length = ox.distance.great_circle_vec(
                        *data['geometry'].coords[0][::-1],
                        *data['geometry'].coords[-1][::-1]
                    )
                else:
                    length = 0
            else:
                length = 0

            if 'width' in data:
                lanes_width_counter += 1
                if isinstance(data["width"], list):
                    width = np.mean([float(x) for x in data["width"]])         
                else:
                    width = float(data["width"])
                        
            elif 'lanes' in data:
                lanes_width_counter += 1
                if isinstance(data["lanes"], list):
                    width = np.mean([int(x) for x in data["lanes"]]) * 3
                else:
                    width = int(data['lanes']) * 3  # przyjmujemy 3 m na pas
            elif 'highway' in data:
                highway_width_counter += 1
                hw = data['highway']
                if isinstance(hw, list):
                    hw = hw[0]
                width = highway_width.get(hw)
            else:
                width = 6

            if "geometry" in data:
                edges_info.append({"id": data["osmid"],'u': u, 'v': v, 'length_m': length, 'width_m': width, "geometry" : data["geometry"]})
        print(f"Roads with 'width' attribute: {lanes_width_counter}\nRoads without 'width' attribute: {highway_width_counter}")

        return edges_info
    

    def convert_to_gdf(self, edges):
        gdf_edges = gpd.GeoDataFrame(edges, crs=self.basic_crs)
        gdf_edges["geometry"] = gdf_edges["geometry"].to_crs(epsg=self.poland_crs)
        return gdf_edges
        