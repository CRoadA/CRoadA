import numpy as np
import osmnx as ox
from shapely import LineString, Polygon
import geopandas as gpd
import networkx as nx

class GraphLoader():
    def __init__(self, poland_crs : int = 2180, basic_crs : int = 4326):
        self.poland_crs = poland_crs
        self.basic_crs = basic_crs


    def load_graph(self, city_input: str) -> nx.MultiDiGraph:
        city_parts = city_input.split(",")
        cleaned_parts = [p.strip() for p in city_parts]
        
        place = {}
        if len(cleaned_parts) > 1:
            place = {"city": cleaned_parts[0], "country": cleaned_parts[1]}
        else:
            place = {"city": cleaned_parts[0]}
        
        try:
            graph = ox.graph_from_place(place, network_type="drive", simplify=False)
        except Exception as e:
            print(f"Error while searching for 'city': ({e}). The next try is searching for 'town'.")
            place["town"] = place.pop("city")
            try:
                graph = ox.graph_from_place(place, network_type="drive", simplify=False)
            except Exception as e:
                print(f"Error while searching for 'town': ({e}). The next try is searching for 'village'.")
                place["village"] = place.pop("town")
                try:
                    graph = ox.graph_from_place(place, network_type="drive", simplify=False)
                except Exception as e:
                    print(f"Error while searching for 'village': ({e})")
                    
        return graph


    def load_graph_from_polygon(self, area : Polygon):
        graph = ox.graph_from_polygon(area, network_type="drive", simplify=False)
        return graph
    

    def get_all_highways(self, graph : nx.MultiDiGraph):
        highways = []
        for u, v, k, data in graph.edges(keys=True, data=True):
            if "highway" in data and data["highway"] not in highways and not isinstance(data["highway"], list):
                highways.append(data["highway"])
            elif isinstance(data["highway"], list):
                highways.extend(h for h in data["highway"] if h not in highways)

        return highways

    def get_avg_width(self, graph : nx.MultiDiGraph, highway : str) -> float:
        count_street = 0
        sum_width = 0

        for u, v, data in list(graph.edges(data=True)):
            width = 0
            if "highway" in data and ((not isinstance(data["highway"], list) and data["highway"] == highway) or (isinstance(data["highway"], list) and highway in data["highway"])):
                if "width" in data:
                    if isinstance(data["width"], list):
                        width = np.mean([self.convert(x, "float") for x in data["width"]])         
                    else:
                        width = self.convert(data["width"], "float")
                elif 'lanes' in data and data["lanes"]:
                    if isinstance(data["lanes"], list):
                        width = np.mean([self.convert(x, "int") for x in data["lanes"]]) * 3
                    else:
                        lanes = self.convert(data["lanes"], "int")
                        if lanes:
                            width = lanes * 3  # przyjmujemy 3 m na pas
                if width is not None:
                    sum_width += width
                    count_street += 1
                
        return sum_width/count_street
    
    def convert(self, data : str | None, type : str = "float"):

        if data is None:
            return None

        data = str(data).strip()
        if data.lower() in ["nan", "none", "null", ""]:
            return None

        if "'" in data:
            try:
                cleaned = data.replace('"', "")
                feet_str, inch_str = cleaned.split("'")
                feet = float(feet_str)
                inches = float(inch_str)
                value_m = feet * 0.3048 + inches * 0.0254  # metry
            except:
                return None

            return int(value_m) if type == "int" else value_m

        if data.endswith("ft") or data.endswith(" ft"):
            try:
                feet = float(data.replace("ft", "").strip())
                value_m = feet * 0.3048
            except:
                return None

            return int(value_m) if type == "int" else value_m

        if data.endswith("m") or data.endswith(" m"):
            try:
                value_m = float(data.replace("m", "").strip())
            except:
                return None

            return int(value_m) if type == "int" else value_m

        try:
            value_m = float(data)
        except:
            return None

        return int(value_m) if type == "int" else value_m



    def get_highways_width(self, graph : nx.MultiDiGraph) -> dict[str, float]:
        highways = self.get_all_highways(graph)
        avg_width = {}

        for highway in highways:
            avg_width[highway] = self.get_avg_width(graph, highway)
        
        return avg_width
    

    def get_edges_measurements(self, graph : nx.MultiDiGraph, residential_max_radius : float) -> list[dict]:

        # mapping - typ drogi : szerokość
        highway_width = self.get_highways_width(graph)

        # small_highways = ["living_street", "residential"]
        small_highways = ["residential"]

        edges_info = []
        lanes_width_counter = 0
        highway_width_counter = 0
        for u, v, k, data in graph.edges(keys=True, data=True):
            name = None
            is_residential = False
            width = None
            if 'length' in data:
                length = data['length']
            elif 'geometry' in data:
                if isinstance(data['geometry'], LineString):
                    length = ox.distance.great_circle_vec(
                        *data['geometry'].coords[0][::-1],
                        *data['geometry'].coords[-1][::-1]
                    )
                else:
                    length = 0
            else:
                length = 0

            if 'width' in data and data["width"]:
                lanes_width_counter += 1
                if isinstance(data["width"], list):
                    width = np.mean([self.convert(x, "float") for x in data["width"]])         
                else:
                    width = self.convert(data["width"], "float")
            else:
                if 'lanes' in data and data["lanes"] and width is None:
                    lanes_width_counter += 1
                    if isinstance(data["lanes"], list):
                        width = np.mean([self.convert(x, "int") for x in data["lanes"]]) * 3
                    else:
                        lanes = self.convert(data["lanes"], "int")
                        if lanes:
                            width = lanes * 3 # przyjmujemy 3 m na pas
                elif 'highway' in data and data["highway"] and width is None:
                    highway_width_counter += 1
                    hw = data['highway']
                    if isinstance(hw, list):
                        hw = hw[0]
                    width = highway_width.get(hw)
                else:
                    width = 6
 
            if "highway" in data and "turning_radius" in data:
                radius = data["turning_radius"]
                hw = data['highway']
                if not isinstance(hw, list) and hw in small_highways and radius < residential_max_radius:
                    is_residential = True
                    
            if "name" in data:
                name = data["name"]

            if "geometry" in data:
                edges_info.append({"id": data["osmid"],'u': u, 'v': v, 'length_m': length, 'width_m': width, "geometry" : data["geometry"], "is_residential" : is_residential, "name" : name, "radius" : data["turning_radius"]})
        print(f"Roads with 'width' attribute: {lanes_width_counter}\nRoads without 'width' attribute: {highway_width_counter}")

        return edges_info
    

    def convert_to_gdf(self, edges : list[dict], new_crs : int | None) -> gpd.GeoDataFrame:
        gdf_edges = gpd.GeoDataFrame(edges, crs=self.basic_crs)
        if new_crs is None:
            new_crs = self.poland_crs

        gdf_edges["geometry"] = gdf_edges["geometry"].to_crs(epsg=new_crs)
        return gdf_edges
    


    def get_graph_for_street(self, graph : nx.MultiDiGraph, street_name : str) -> nx.MultiDiGraph:
        # lista krawędzi, które pasują do nazwy
        selected_edges = []

        for u, v, k, data in graph.edges(keys=True, data=True):
            name_attr = data.get("name")
            match = False
            
            if isinstance(name_attr, list):
                if street_name in name_attr:
                    match = True
            elif name_attr == street_name:
                match = True
                
            if match:
                selected_edges.append((u, v, k))

        subgraph = graph.edge_subgraph(selected_edges).copy()
        return subgraph
        