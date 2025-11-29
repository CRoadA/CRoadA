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

        first_node_id = next(iter(G.nodes()))
        x = G.nodes[first_node_id].get('x', 0)

        # If X is between -180 and 180, it's likely WGS84 (degrees).
        if -180 < x < 180:
            print("  [INFO] Coordinates in degrees detected. Projecting to UTM...")
            try:
                G_proj = ox.project_graph(G)
                return G_proj
            except Exception as e:
                print(f"  [ERROR] Failed to project graph: {e}")
                return G
        else:
            print("  [INFO] Graph is already in metric system.")
            return G

    def _calculate_circumradius(self, p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # Side lengths
        a = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        b = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
        c = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

        # Ignore if points are too close
        if a < 0.1 or b < 0.1 or c < 0.1:
            return float('inf')

        s = (a + b + c) / 2
        area_sq = s * (s - a) * (s - b) * (s - c)

        # Error tolerance
        if area_sq <= 1e-12:
            return float('inf')

        area = math.sqrt(area_sq)

        # Radius R = abc / 4A
        try:
            R = (a * b * c) / (4 * area)
            return R
        except ZeroDivisionError:
            return float('inf')

    def analyze_curvature(self, max_radius=500.0):
        street_radii = []
        junction_radii = []

        print("  [INFO] Starting geometry analysis...")

        # 1. Edges (Streets)
        edges_with_geom = 0
        for u, v, k, data in self.G.edges(keys=True, data=True):
            if 'geometry' in data and isinstance(data['geometry'], LineString):
                coords = list(data['geometry'].coords)
                if len(coords) > 2:
                    edges_with_geom += 1
                    for i in range(len(coords) - 2):
                        R = self._calculate_circumradius(coords[i], coords[i + 1], coords[i + 2])
                        if R < max_radius:
                            street_radii.append(R)

        print(f"  [DEBUG] Found {edges_with_geom} edges with detailed geometry.")

        # 2. Junctions
        nodes_checked = 0
        for node in self.G.nodes():
            nodes_checked += 1
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


def test_krakow():
    print("--- START ANALYSIS FOR KRAKOW ---")
    start_time = time.time()
    place_name = "Kraków, Poland"
    print(f"1. Downloading graph: {place_name}")

    try:
        G = ox.graph_from_place(place_name, network_type="drive")
    except Exception as e:
        print(f"Download error: {e}")
        return

    print(f"   Downloaded! Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")

    analyzer = CurvatureAnalyzer(G)
    stats = analyzer.analyze_curvature(max_radius=300.0)

    streets = stats['street_curvature']
    junctions = stats['junction_turns']

    print(f"\n--- RESULTS ---")
    if streets:
        print(f"Streets (segments): {len(streets)} samples")
        print(f"  -> Mean: {np.mean(streets):.2f} m")
        print(f"  -> Median: {np.median(streets):.2f} m")
    else:
        print("Streets: No data (segments too straight or area too small?)")

    if junctions:
        print(f"Junctions: {len(junctions)} samples")
        print(f"  -> Mean: {np.mean(junctions):.2f} m")
        print(f"  -> Median: {np.median(junctions):.2f} m")
    else:
        print("Junctions: No data")

    if streets or junctions:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        if streets:
            axes[0].hist(streets, bins=30, color='skyblue', edgecolor='black', density=True)
            axes[0].set_title(f'Street Curvature (Internal)\nn={len(streets)}')

        if junctions:
            axes[1].hist(junctions, bins=30, color='salmon', edgecolor='black', density=True)
            axes[1].set_title(f'Junction Turn Radii\nn={len(junctions)}')

        plt.tight_layout()
        plt.show()
    else:
        print("No data to plot.")

    print(f"Total time: {time.time() - start_time:.2f} s")


if __name__ == "__main__":
    test_krakow()