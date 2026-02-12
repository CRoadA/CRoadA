import numpy as np
import networkx as nx
import math
import warnings
import traceback
import os
import matplotlib.pyplot as plt
from shapely.geometry import LineString, box
from typing import Tuple
from skimage.morphology import remove_small_objects, binary_closing, disk
from scipy.spatial import cKDTree

from grid_manager import GridManager, GRID_INDICES
from graph_remaker.morphological_remaker import discover_streets

warnings.filterwarnings("ignore", category=FutureWarning)

# --- CONFIGURATION ---
OVERLAP = 10
SNAP_DIST = 5.0
MIN_SIZE = 10
DOWNSAMPLING = 1

DEBUG_DIR = "debug_segments"


class LargeGridProcessor:
    def __init__(self, grid_manager: GridManager):
        self.gm = grid_manager
        self.meta = grid_manager.get_metadata()
        self.G = nx.MultiDiGraph()
        self.G.graph['crs'] = 'EPSG:4326'

        if not os.path.exists(DEBUG_DIR):
            os.makedirs(DEBUG_DIR)
        else:
            for f in os.listdir(DEBUG_DIR):
                try:
                    os.remove(os.path.join(DEBUG_DIR, f))
                except OSError:
                    pass

    def run(self) -> nx.MultiDiGraph:
        rows_total = self.meta.rows_number
        cols_total = self.meta.columns_number
        seg_h = self.meta.segment_h
        seg_w = self.meta.segment_w

        n_rows = math.ceil(rows_total / seg_h)
        n_cols = math.ceil(cols_total / seg_w)

        print(f"Processing Grid: {rows_total}x{cols_total}. Segments: {n_rows}x{n_cols}.")

        processed = 0
        skipped = 0

        for r in range(n_rows):
            for c in range(n_cols):
                core_y0 = r * seg_h
                core_x0 = c * seg_w
                core_y1 = min(core_y0 + seg_h, rows_total)
                core_x1 = min(core_x0 + seg_w, cols_total)

                try:
                    grid, offset_y, offset_x = self._load_super_segment(r, c)
                    mask = grid[:, :, GRID_INDICES.IS_STREET] > 0

                    if np.sum(mask) == 0:
                        processed += 1
                        continue

                    mask = remove_small_objects(mask, min_size=MIN_SIZE)
                    mask = binary_closing(mask, footprint=disk(1))
                    grid[:, :, GRID_INDICES.IS_STREET] = mask.astype(np.float32)

                    try:
                        res = discover_streets(grid)
                        all_crossroads = res[0] + res[1]
                        all_streets = res[2] + res[3]

                        self._add_to_graph_clipped(
                            all_crossroads,
                            all_streets,
                            global_offset_y=offset_y,
                            global_offset_x=offset_x,
                            clip_bounds=(core_y0, core_x0, core_y1, core_x1),
                            segment_id=(r, c)
                        )

                    except Exception:
                        print(f"CRITICAL ALGORITHM ERROR in segment ({r}, {c}):")
                        traceback.print_exc()
                        skipped += 1

                except Exception as e:
                    print(f"CRITICAL: I/O Error in segment ({r}, {c}): {e}")
                    traceback.print_exc()
                    skipped += 1

                processed += 1
                if processed % 10 == 0:
                    print(f"Processed segment ({r}, {c})...")

        print(f"Processing finished. Starting Snapping...")
        return self._finalize()

    def _load_super_segment(self, r, c) -> Tuple[np.ndarray, int, int]:
        seg_h = self.meta.segment_h
        seg_w = self.meta.segment_w

        global_y0 = r * seg_h
        global_x0 = c * seg_w

        read_y0 = max(0, global_y0 - OVERLAP)
        read_x0 = max(0, global_x0 - OVERLAP)
        read_y1 = min(self.meta.rows_number, min(global_y0 + seg_h, self.meta.rows_number) + OVERLAP)
        read_x1 = min(self.meta.columns_number, min(global_x0 + seg_w, self.meta.columns_number) + OVERLAP)

        target_h = read_y1 - read_y0
        target_w = read_x1 - read_x0

        super_grid = np.zeros((target_h, target_w, 3), dtype=np.float32)

        start_seg_r = read_y0 // seg_h
        end_seg_r = (read_y1 - 1) // seg_h
        start_seg_c = read_x0 // seg_w
        end_seg_c = (read_x1 - 1) // seg_w

        for sr in range(start_seg_r, end_seg_r + 1):
            for sc in range(start_seg_c, end_seg_c + 1):
                chunk = self.gm.read_segment(sr, sc)
                ch_h, ch_w, _ = chunk.shape

                chunk_glob_y0 = sr * seg_h
                chunk_glob_x0 = sc * seg_w

                iy0 = max(read_y0, chunk_glob_y0)
                ix0 = max(read_x0, chunk_glob_x0)
                iy1 = min(read_y1, chunk_glob_y0 + ch_h)
                ix1 = min(read_x1, chunk_glob_x0 + ch_w)

                if iy1 > iy0 and ix1 > ix0:
                    src_y = iy0 - chunk_glob_y0
                    src_x = ix0 - chunk_glob_x0
                    dst_y = iy0 - read_y0
                    dst_x = ix0 - read_x0
                    h = iy1 - iy0
                    w = ix1 - ix0
                    super_grid[dst_y: dst_y + h, dst_x: dst_x + w] = \
                        chunk[src_y: src_y + h, src_x: src_x + w]

        return super_grid, read_y0, read_x0

    def _add_to_graph_clipped(self, crossroads, streets, global_offset_y, global_offset_x, clip_bounds, segment_id):
        """Dodaje elementy do grafu przycinając je do granic segmentu (core bounds)."""
        core_y0, core_x0, core_y1, core_x1 = clip_bounds
        EPSILON = 0.1
        clip_box = box(core_x0 - EPSILON, core_y0 - EPSILON, core_x1 + EPSILON, core_y1 + EPSILON)
        point_map = {}

        for cr in crossroads:
            if not cr.points: continue
            ys = [p[0] for p in cr.points]
            xs = [p[1] for p in cr.points]
            cy = sum(ys) / len(ys)
            cx = sum(xs) / len(xs)
            gy = global_offset_y + cy
            gx = global_offset_x + cx

            if not (core_y0 <= gy < core_y1 and core_x0 <= gx < core_x1):
                continue

            node_id = f"{gy:.1f}_{gx:.1f}"
            if not self.G.has_node(node_id):
                self.G.add_node(node_id, y=gy, x=gx, type='crossroad')

            for p in cr.points:
                point_map[p] = node_id

        # --- ULICE ---
        for st in streets:
            if len(st.linestring) < 2: continue

            slope_val = getattr(st, 'max_longitudinal_slope', 0.0)
            width_val = getattr(st, 'width', 0.0)

            full_geo_points = []
            for (ly, lx) in st.linestring:
                full_geo_points.append((global_offset_x + lx, global_offset_y + ly))

            full_geom = LineString(full_geo_points)

            clipped = full_geom.intersection(clip_box)

            if clipped.is_empty:
                continue

            geoms_to_process = []
            if clipped.geom_type == 'LineString':
                geoms_to_process.append(clipped)
            elif clipped.geom_type == 'MultiLineString':
                geoms_to_process.extend(clipped.geoms)
            elif clipped.geom_type == 'GeometryCollection':
                for g in clipped.geoms:
                    if g.geom_type == 'LineString':
                        geoms_to_process.append(g)

            for geom in geoms_to_process:
                if len(geom.coords) < 2: continue
                coords = list(geom.coords)

                if len(coords) > 2:
                    coords = [coords[0]] + coords[1:-1:DOWNSAMPLING] + [coords[-1]]

                start_x, start_y = coords[0]
                end_x, end_y = coords[-1]

                loc_start_y = int(round(start_y - global_offset_y))
                loc_start_x = int(round(start_x - global_offset_x))
                loc_end_y = int(round(end_y - global_offset_y))
                loc_end_x = int(round(end_x - global_offset_x))

                u = point_map.get((loc_start_y, loc_start_x))
                if not u:
                    u = f"{start_y:.1f}_{start_x:.1f}"
                    if not self.G.has_node(u): self.G.add_node(u, y=start_y, x=start_x, type='connector')

                v = point_map.get((loc_end_y, loc_end_x))
                if not v:
                    v = f"{end_y:.1f}_{end_x:.1f}"
                    if not self.G.has_node(v): self.G.add_node(v, y=end_y, x=end_x, type='connector')

                if u != v:
                    self.G.add_edge(u, v,
                                    geometry=LineString(coords),
                                    slope=float(slope_val),
                                    width=float(width_val))

    def _finalize(self):
        """Finalizuje graf: łączy bliskie węzły (snapping) i konwertuje na Lat/Lon."""
        if len(self.G.nodes) == 0:
            return self.G

        nodes = list(self.G.nodes(data=True))
        coords = np.array([[d['y'], d['x']] for n, d in nodes])
        ids = [n for n, d in nodes]

        print(f"Snapping {len(nodes)} nodes with radius {SNAP_DIST}...")
        tree = cKDTree(coords)
        pairs = list(tree.query_pairs(r=SNAP_DIST))

        if pairs:
            print(f"  Found {len(pairs)} pairs to merge.")
            uf = nx.utils.UnionFind()
            for i, j in pairs:
                uf.union(ids[i], ids[j])

            new_G = nx.MultiDiGraph()
            new_G.graph = self.G.graph.copy()
            winners = {}

            for component in uf.to_sets():
                winner_node = \
                sorted(list(component), key=lambda n: (0 if self.G.nodes[n].get('type') == 'crossroad' else 1, n))[0]

                if winner_node not in new_G:
                    new_G.add_node(winner_node, **self.G.nodes[winner_node])

                for node in component:
                    winners[node] = winner_node

            for u, v, key, data in self.G.edges(keys=True, data=True):
                final_u = winners.get(u, u)
                final_v = winners.get(v, v)

                if final_u not in new_G: new_G.add_node(final_u, **self.G.nodes[final_u])
                if final_v not in new_G: new_G.add_node(final_v, **self.G.nodes[final_v])

                if final_u != final_v:
                    old_geom = data.get('geometry')
                    if old_geom:
                        g_coords = list(old_geom.coords)
                        uy, ux = new_G.nodes[final_u]['y'], new_G.nodes[final_u]['x']
                        vy, vx = new_G.nodes[final_v]['y'], new_G.nodes[final_v]['x']

                        new_g_coords = [(ux, uy)] + g_coords[1:-1] + [(vx, vy)]
                        data['geometry'] = LineString(new_g_coords)

                    new_G.add_edge(final_u, final_v, key=key, **data)

            self.G = new_G

        print("Converting to Lat/Lon...")
        lat0 = self.meta.upper_left_latitude
        lon0 = self.meta.upper_left_longitude
        res = self.meta.grid_density
        m_lat = 111132.0
        m_lon = 111412.0 * math.cos(math.radians(lat0))

        for n, d in self.G.nodes(data=True):
            d['y'] = lat0 - (d['y'] * res / m_lat)
            d['x'] = lon0 + (d['x'] * res / m_lon)

        for u, v, k, d in self.G.edges(keys=True, data=True):
            if 'geometry' in d:
                new_coords = []
                for x_px, y_px in d['geometry'].coords:
                    lat = lat0 - (y_px * res / m_lat)
                    lon = lon0 + (x_px * res / m_lon)
                    new_coords.append((lon, lat))
                d['geometry'] = LineString(new_coords)

        return self.G


def process_large_grid(file_name: str, output_graph_path: str = None) -> nx.MultiDiGraph:
    gm = GridManager(file_name)
    proc = LargeGridProcessor(gm)
    G = proc.run()
    if output_graph_path:
        nx.write_graphml(G, output_graph_path)
    return G