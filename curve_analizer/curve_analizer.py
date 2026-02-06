import osmnx as ox
import networkx as nx
import time
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import math
import warnings

# Ignore runtime warnings (handled manually)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class CurvatureAnalyzer:
    def __init__(self, G: nx.MultiDiGraph):
        self.G = self._ensure_projected(G)

    def _ensure_projected(self, G: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        Reliably check if graph is in meters.
        """
        if not G.nodes:
            return G

        if G.graph.get('crs') == 'EPSG:4326':
            print("  [INFO] Graph CRS is EPSG:4326. Projecting to UTM...")
            try:
                G_proj = ox.project_graph(G)
                return G_proj
            except Exception as e:
                print(f"  [ERROR] Failed to project graph via osmnx: {e}. Analyzing as is.")
                return G

        first_node_id = next(iter(G.nodes()))
        x = G.nodes[first_node_id].get('x', 0)

        if -180 < x < 180:
            print("  [INFO] Coordinates in degrees detected based on values. Projecting to UTM...")
            try:
                G_proj = ox.project_graph(G)
                return G_proj
            except Exception as e:
                print(f"  [ERROR] Failed to project graph: {e}")
                return G
        else:
            print("  [INFO] Graph is likely already in metric system.")
            return G

    def _calculate_circumradius(self, p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # Side lengths
        a = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        b = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
        c = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

        if a < 0.1 or b < 0.1 or c < 0.1:
            return float('inf')

        s = (a + b + c) / 2
        area_sq = s * (s - a) * (s - b) * (s - c)

        if area_sq <= 1e-12:
            return float('inf')

        area = math.sqrt(area_sq)

        try:
            R = (a * b * c) / (4 * area)
            return R
        except ZeroDivisionError:
            return float('inf')

    def analyze_curvature(self, max_radius=500.0):
        street_radii = []
        junction_radii = []

        edges_with_geom = 0
        for u, v, k, data in self.G.edges(keys=True, data=True):
            if 'geometry' in data and isinstance(data['geometry'], LineString):
                coords = list(data['geometry'].coords)
                if len(coords) > 2:
                    edges_with_geom += 1
                    local_radii = []
                    for i in range(len(coords) - 2):
                        R = self._calculate_circumradius(coords[i], coords[i + 1], coords[i + 2])
                        if R < max_radius:
                            local_radii.append(R)

                    street_radii.extend(local_radii)

        # 2. Junctions
        for node in self.G.nodes():
            in_edges = list(self.G.in_edges(node, data=True))
            out_edges = list(self.G.out_edges(node, data=True))

            for u_in, _, data_in in in_edges:
                if u_in == node: continue

                if 'geometry' in data_in and len(data_in['geometry'].coords) >= 2:
                    pt_a = list(data_in['geometry'].coords)[-2]
                else:
                    pt_a = (self.G.nodes[u_in]['x'], self.G.nodes[u_in]['y'])

                pt_b = (self.G.nodes[node]['x'], self.G.nodes[node]['y'])

                for _, v_out, data_out in out_edges:
                    if v_out == node or u_in == v_out: continue

                    if 'geometry' in data_out and len(data_out['geometry'].coords) >= 2:
                        pt_c = list(data_out['geometry'].coords)[1]
                    else:
                        pt_c = (self.G.nodes[v_out]['x'], self.G.nodes[v_out]['y'])

                    R = self._calculate_circumradius(pt_a, pt_b, pt_c)

                    if R < max_radius:
                        if self.G.degree(node) <= 2:
                            street_radii.append(R)
                        else:
                            junction_radii.append(R)

        return {'street_curvature': street_radii, 'junction_turns': junction_radii}