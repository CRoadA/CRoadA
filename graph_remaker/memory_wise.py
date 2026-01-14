import numpy as np
import networkx as nx
import math
import warnings
from shapely.geometry import LineString
from typing import List, Tuple, Dict
from skimage.morphology import remove_small_objects, binary_closing, disk
from scipy.spatial import cKDTree

from grid_manager import GridManager, GRID_INDICES
from graph_remaker.morphological_remaker import discover_streets

# Suppress future warnings from dependencies
warnings.filterwarnings("ignore", category=FutureWarning)

# --- CONFIGURATION ---
OVERLAP = 10  # Margin (px) to capture context from neighboring segments
SNAP_DIST = 5.0  # Distance threshold (px) for merging close nodes
MIN_SIZE = 10  # Minimum object size for noise removal
DOWNSAMPLING = 5  # Step size for linestring simplification


class LargeGridProcessor:
    """
    Handles processing of large grid files by dividing them into segments
    with overlap to ensure continuity, then stitches the results into a single graph.
    """

    def __init__(self, grid_manager: GridManager):
        self.gm = grid_manager
        self.meta = grid_manager.get_metadata()

        # Initialize global graph with WGS84 CRS
        self.G = nx.MultiDiGraph()
        self.G.graph['crs'] = 'EPSG:4326'

    def run(self) -> nx.MultiDiGraph:
        """
        Main execution loop. Iterates over grid segments, processes them,
        and aggregates results into the global graph.
        """
        rows_total = self.meta.rows_number
        cols_total = self.meta.columns_number
        seg_h = self.meta.segment_h
        seg_w = self.meta.segment_w

        # Calculate grid dimensions in segments
        n_rows = math.ceil(rows_total / seg_h)
        n_cols = math.ceil(cols_total / seg_w)

        print(f"Starting processing. Grid: {rows_total}x{cols_total}. Segments: {n_rows}x{n_cols}.")

        processed = 0
        skipped = 0

        for r in range(n_rows):
            for c in range(n_cols):
                try:
                    # 1. Build "Super-Segment" (Central segment + Overlap from neighbors)
                    # This ensures seamless detection across borders.
                    grid, offset_y, offset_x = self._load_super_segment(r, c)

                    # 2. Data Cleaning
                    # Use binary_closing to bridge small gaps without excessive thickening.
                    mask = grid[:, :, GRID_INDICES.IS_STREET] > 0

                    if np.sum(mask) == 0:
                        processed += 1
                        continue

                    mask = remove_small_objects(mask, min_size=MIN_SIZE)
                    mask = binary_closing(mask, footprint=disk(1))

                    grid[:, :, GRID_INDICES.IS_STREET] = mask.astype(np.float32)

                    # 3. Execute Street Discovery
                    try:
                        (cl_cr, c_cr, cl_st, c_st) = discover_streets(grid)

                        # 4. Add to Graph (Coordinate Translation)
                        # Offsets are subtracted to map local super-segment coords to global.
                        self._add_to_graph(
                            cl_cr + c_cr,
                            cl_st + c_st,
                            global_offset_y=offset_y,
                            global_offset_x=offset_x
                        )

                    except IndexError:
                        # Geometric complexity might cause algorithm failure in rare cases
                        print(f"WARN: Geometry error in segment ({r}, {c})")
                        skipped += 1
                    except Exception as e:
                        print(f"WARN: Algorithm error in segment ({r}, {c}): {e}")
                        skipped += 1

                except Exception as e:
                    print(f"CRITICAL: I/O Error in segment ({r}, {c}): {e}")
                    skipped += 1

                processed += 1
                if processed % 10 == 0:
                    print(f"Progress: {processed}/{n_rows * n_cols}...")

        print(f"Processing finished. Skipped: {skipped}. Starting topology repair (Snapping)...")
        return self._finalize()

    def _load_super_segment(self, r, c) -> Tuple[np.ndarray, int, int]:
        """
        Loads the target segment (r, c) and stitches overlapping margins from neighbors.
        Returns: (Padded Grid, Global Y Start, Global X Start)
        """
        seg_h = self.meta.segment_h
        seg_w = self.meta.segment_w

        # 1. Determine bounds of the "Super-Segment" in global coordinates
        # Center segment bounds
        global_y0 = r * seg_h
        global_x0 = c * seg_w
        global_y1 = min(global_y0 + seg_h, self.meta.rows_number)
        global_x1 = min(global_x0 + seg_w, self.meta.columns_number)

        # Bounds with overlap (clamped to image dimensions)
        read_y0 = max(0, global_y0 - OVERLAP)
        read_x0 = max(0, global_x0 - OVERLAP)
        read_y1 = min(self.meta.rows_number, global_y1 + OVERLAP)
        read_x1 = min(self.meta.columns_number, global_x1 + OVERLAP)

        # Target dimensions
        target_h = read_y1 - read_y0
        target_w = read_x1 - read_x0

        # Initialize container
        super_grid = np.zeros((target_h, target_w, 3), dtype=np.float32)

        # 2. Tiling Logic
        # Identify which physical segments intersect with the requested area
        start_seg_r = read_y0 // seg_h
        end_seg_r = (read_y1 - 1) // seg_h

        start_seg_c = read_x0 // seg_w
        end_seg_c = (read_x1 - 1) // seg_w

        # Iterate over relevant segments and stitch them
        for sr in range(start_seg_r, end_seg_r + 1):
            for sc in range(start_seg_c, end_seg_c + 1):
                # Load source chunk
                chunk = self.gm.read_segment(sr, sc)
                ch_h, ch_w, _ = chunk.shape

                # Global coords of the chunk
                chunk_glob_y0 = sr * seg_h
                chunk_glob_x0 = sc * seg_w
                chunk_glob_y1 = chunk_glob_y0 + ch_h
                chunk_glob_x1 = chunk_glob_x0 + ch_w

                # Calculate Intersection
                iy0 = max(read_y0, chunk_glob_y0)
                ix0 = max(read_x0, chunk_glob_x0)
                iy1 = min(read_y1, chunk_glob_y1)
                ix1 = min(read_x1, chunk_glob_x1)

                if iy1 > iy0 and ix1 > ix0:
                    # Source indices (relative to chunk)
                    src_y = iy0 - chunk_glob_y0
                    src_x = ix0 - chunk_glob_x0

                    # Destination indices (relative to super_grid)
                    dst_y = iy0 - read_y0
                    dst_x = ix0 - read_x0

                    height = iy1 - iy0
                    width = ix1 - ix0

                    super_grid[dst_y: dst_y + height, dst_x: dst_x + width] = \
                        chunk[src_y: src_y + height, src_x: src_x + width]

        return super_grid, read_y0, read_x0

    def _add_to_graph(self, crossroads, streets, global_offset_y, global_offset_x):
        """Adds discovered entities to the graph with global coordinates."""

        # Mapping: Local Point -> Global Node ID
        point_map = {}

        # Process Crossroads
        for cr in crossroads:
            if not cr.points: continue

            # Centroid calculation
            ys = [p[0] for p in cr.points]
            xs = [p[1] for p in cr.points]
            cy = sum(ys) / len(ys)
            cx = sum(xs) / len(xs)

            # Global coordinates
            gy = global_offset_y + cy
            gx = global_offset_x + cx

            # ID as string (will be snapped later)
            node_id = f"{gy:.1f}_{gx:.1f}"

            if not self.G.has_node(node_id):
                self.G.add_node(node_id, y=gy, x=gx, type='crossroad')

            for p in cr.points:
                point_map[p] = node_id

        # Process Streets
        for st in streets:
            if len(st.linestring) < 2: continue

            # Order and simplify geometry
            clean_pts = self._sort_and_downsample(st.linestring)
            if len(clean_pts) < 2: continue

            # Start/End points (local)
            start_p = clean_pts[0]
            end_p = clean_pts[-1]

            # Resolve Nodes
            u = self._get_node_id(start_p, global_offset_y, global_offset_x, point_map)
            v = self._get_node_id(end_p, global_offset_y, global_offset_x, point_map)

            if u != v:
                # Global geometry construction
                geo_points = []
                for (ly, lx) in clean_pts:
                    geo_points.append((global_offset_x + lx, global_offset_y + ly))

                geom = LineString(geo_points)
                self.G.add_edge(u, v, geometry=geom)

    def _get_node_id(self, local_p, off_y, off_x, mapping):
        """Retrieves or creates a node ID. Creates a connector node if not a crossroad."""
        if local_p in mapping:
            return mapping[local_p]

        gy = off_y + local_p[0]
        gx = off_x + local_p[1]
        node_id = f"{gy:.1f}_{gx:.1f}"

        if not self.G.has_node(node_id):
            self.G.add_node(node_id, y=gy, x=gx, type='connector')

        return node_id

    def _sort_and_downsample(self, points):
        """Sorts points using Nearest Neighbor heuristic and downsamples."""
        if not points: return []
        ordered = [points[0]]
        remain = set(points[1:])
        curr = points[0]

        while remain:
            # Optimization: Search only in local vicinity
            candidates = [p for p in remain if abs(p[0] - curr[0]) < 15 and abs(p[1] - curr[1]) < 15]
            if not candidates: candidates = remain

            nearest = min(candidates, key=lambda p: (p[0] - curr[0]) ** 2 + (p[1] - curr[1]) ** 2)
            ordered.append(nearest)
            remain.remove(nearest)
            curr = nearest

        return ordered[::DOWNSAMPLING] + [ordered[-1]]

    def _finalize(self):
        """Performs Snapping and Coordinate Conversion."""
        if len(self.G.nodes) == 0:
            return self.G

        # 1. Snapping (Merging close nodes)
        nodes = list(self.G.nodes(data=True))
        coords = np.array([[d['y'], d['x']] for n, d in nodes])
        ids = [n for n, d in nodes]

        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=SNAP_DIST)

        # Use UnionFind to group nodes to be merged
        uf = nx.utils.UnionFind()
        for i, j in pairs:
            uf.union(ids[i], ids[j])

        # Merge groups
        for component in uf.to_sets():
            if len(component) < 2: continue
            nodes_list = list(component)

            # Select winner (Prefer Crossroad)
            winner = sorted(nodes_list, key=lambda n: 0 if self.G.nodes[n].get('type') == 'crossroad' else 1)[0]

            for loser in nodes_list:
                if loser == winner: continue

                # Move edges
                for u, _, d in list(self.G.in_edges(loser, data=True)):
                    if u != winner: self.G.add_edge(u, winner, **d)
                for _, v, d in list(self.G.out_edges(loser, data=True)):
                    if v != winner: self.G.add_edge(winner, v, **d)

                self.G.remove_node(loser)

        # 2. Georeferencing (Pixel -> Lat/Lon)
        lat0 = self.meta.upper_left_latitude
        lon0 = self.meta.upper_left_longitude
        res = self.meta.grid_density

        # Approximation for meters per degree
        m_lat = 111132.0
        m_lon = 111412.0 * math.cos(math.radians(lat0))

        for n, d in self.G.nodes(data=True):
            self.G.nodes[n]['y'] = lat0 - (d['y'] * res / m_lat)
            self.G.nodes[n]['x'] = lon0 + (d['x'] * res / m_lon)

        return self.G


def process_large_grid(file_name: str, output_graph_path: str = None) -> nx.MultiDiGraph:
    """Wrapper function to instantiate processor and run the pipeline."""
    gm = GridManager(file_name)
    proc = LargeGridProcessor(gm)
    G = proc.run()
    if output_graph_path:
        nx.write_graphml(G, output_graph_path)
    return G