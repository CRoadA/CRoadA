import osmnx as ox
import networkx as nx
import numpy as np
import math
import warnings
from shapely.geometry import LineString, Point

warnings.filterwarnings("ignore", category=RuntimeWarning)


class CurvatureAnalyzer:
    def __init__(self, G: nx.MultiDiGraph):
        self.G = G.copy()

        first_node = next(iter(self.G.nodes(data=True)))
        x_val = first_node[1].get('x', 0)

        self.is_degrees = -180 < x_val < 180

        if self.is_degrees:
            print("  [Curvature] Wykryto współrzędne geograficzne (stopnie).")
            try:
                self.G = ox.project_graph(self.G)
                self.is_degrees = False
                print("  [Curvature] Pomyślnie zrzutowano graf na metry (UTM).")
            except Exception as e:
                print(f"  [Curvature] Ostrzeżenie: Nie udało się zrzutować grafu ({e}).")
                print("  [Curvature] Będę dokonywać przybliżonej konwersji stopni na metry w locie.")

    def _calculate_circumradius(self, p1, p2, p3):
        """Oblicza promień okręgu opisanego na 3 punktach."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # Długości boków
        a = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        b = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
        c = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

        if a < 1e-9 or b < 1e-9 or c < 1e-9: return float('inf')

        s = (a + b + c) / 2
        area_sq = s * (s - a) * (s - b) * (s - c)

        if area_sq <= 1e-16: return float('inf')

        try:
            R = (a * b * c) / (4 * math.sqrt(area_sq))
            return R
        except (ValueError, ZeroDivisionError):
            return float('inf')

    def analyze_curvature(self, max_radius=1000.0):
        street_radii = []
        junction_radii = []

        if not self.G.nodes:
            return {'street_curvature': [], 'junction_turns': []}

        print("  [Curvature] Rozpoczynam analizę węzłów...")

        G_undir = self.G.to_undirected()

        debug_count = 0

        for node in self.G.nodes():
            neighbors = list(G_undir.neighbors(node))
            if len(neighbors) < 2: continue

            pt_center = (self.G.nodes[node]['x'], self.G.nodes[node]['y'])

            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    n1 = neighbors[i]
                    n2 = neighbors[j]

                    LOOKAHEAD = 10.0
                    if self.is_degrees:
                        LOOKAHEAD = 0.00009

                    pt_arm1 = self._get_interpolated_point(node, n1, distance=LOOKAHEAD)
                    pt_arm2 = self._get_interpolated_point(node, n2, distance=LOOKAHEAD)

                    if self.is_degrees:
                        lat_ref = pt_center[1]
                        m_per_deg_lat = 111132.0
                        m_per_deg_lon = 111412.0 * math.cos(math.radians(lat_ref))

                        pA = ((pt_arm1[0] - pt_center[0]) * m_per_deg_lon, (pt_arm1[1] - pt_center[1]) * m_per_deg_lat)
                        pB = (0.0, 0.0)
                        pC = ((pt_arm2[0] - pt_center[0]) * m_per_deg_lon, (pt_arm2[1] - pt_center[1]) * m_per_deg_lat)

                        R = self._calculate_circumradius(pA, pB, pC)
                    else:
                        R = self._calculate_circumradius(pt_arm1, pt_center, pt_arm2)

                    if R < 100 and debug_count < 3:
                        print(f"    DEBUG: Węzeł {node}: Wykryto R = {R:.2f} m")
                        debug_count += 1

                    if self.G.degree(node) <= 2:
                        street_radii.append(R)
                    else:
                        junction_radii.append(R)

        return {'street_curvature': street_radii, 'junction_turns': junction_radii}

    def _get_interpolated_point(self, node_start, node_end, distance=10.0):
        """Pobiera punkt na krawędzi w zadanej odległości od startu."""
        edge_data = self.G.get_edge_data(node_start, node_end)
        if not edge_data: edge_data = self.G.get_edge_data(node_end, node_start)
        data = edge_data[0]

        geom = data.get('geometry')
        if not geom or not isinstance(geom, LineString):
            p1 = (self.G.nodes[node_start]['x'], self.G.nodes[node_start]['y'])
            p2 = (self.G.nodes[node_end]['x'], self.G.nodes[node_end]['y'])
            geom = LineString([p1, p2])

        p_start = (self.G.nodes[node_start]['x'], self.G.nodes[node_start]['y'])
        geom_start = geom.coords[0]

        dist_start = math.hypot(geom_start[0] - p_start[0], geom_start[1] - p_start[1])

        used_dist = distance
        if geom.length < distance:
            used_dist = geom.length * 0.5

        if dist_start < 1e-5:
            pt = geom.interpolate(used_dist)
        else:
            pt = geom.interpolate(geom.length - used_dist)

        return (pt.x, pt.y)