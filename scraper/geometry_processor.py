import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point
import geopandas as gpd
import math

class GeometryProcessor():
    def get_straight_line_coefficients(self, x1, y1, x2, y2):
        if x1 == x2 and y1 == y2:
            raise ValueError("The points are the same — cannot determine the line.")

        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1

        return (a, b, c)


    def get_perpendicular_line(self, a, b):
        if a == 0 and b == 0:
            raise ValueError("Incorrect coefficients — this is not the equation of a line.")

        a_p = b
        b_p = -a
        return (a_p, b_p)


    def segment_from_line(self, A, B, x_s, y_s, width):
        vx, vy = B, -A

        # normalization
        norm = np.sqrt(A**2 + B**2)
        vx /= norm
        vy /= norm

        d = width / 2

        # line ends
        x1 = x_s + vx * d
        y1 = y_s + vy * d
        x2 = x_s - vx * d
        y2 = y_s - vy * d

        return (x1, y1), (x2, y2)


    def get_edge_polygon(self, edge):
        xs, ys = edge.geometry.xy
        xs = list(xs)
        ys = list(ys)

        left_side = []
        right_side = []

        if len(xs) != len(ys):
            return None

        for i in range(len(xs) - 1):
            x1, y1 = xs[i], ys[i]
            x2, y2 = xs[i+1], ys[i+1]

            # skip the line which length = 0
            if x1 == x2 and y1 == y2:
                continue

            coef = self.get_straight_line_coefficients(x1, y1, x2, y2)
            perp = self.get_perpendicular_line(coef[0], coef[1])

            try:
                (lx1, ly1), (rx1, ry1) = self.segment_from_line(perp[0], perp[1], x1, y1, edge.width_m)
                (lx2, ly2), (rx2, ry2) = self.segment_from_line(perp[0], perp[1], x2, y2, edge.width_m)
            except Exception:
                return None

            left_side.extend([(lx1, ly1), (lx2, ly2)])
            right_side.extend([(rx1, ry1), (rx2, ry2)])

        # there are no points
        if len(left_side) < 2 or len(right_side) < 2:
            return None

        coords = left_side + right_side[::-1]
        try:
            polygon = Polygon(coords)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
        except Exception:
            return None

        return polygon



    def is_residential(self, gdf_edges: gpd.GeoDataFrame, coordinates : tuple[int, int]) -> bool:
        point = Point(coordinates)

        columns = gdf_edges.columns.to_list()

        if "geometry" not in columns or "is_residential" not in columns:
            return False

        for _, row in gdf_edges.iterrows():
            geometry = row["geometry"]
            if isinstance(geometry, Polygon) and geometry.covers(point):
                print(f"Geometry processor - is_residential: {bool(row["is_residential"])}")
                return bool(row["is_residential"])

        return False
    

    def is_residential_fast(self, gdf_edges: gpd.GeoDataFrame, coordinates: tuple[int, int]) -> bool:
        columns = gdf_edges.columns.to_list()

        if "geometry" not in columns or "is_residential" not in columns:
            return False

        point = Point(coordinates)

        matches_mask = gdf_edges.geometry.covers(point)
        matching_rows = gdf_edges[matches_mask]

        if matching_rows.empty:
            return False

        return bool(matching_rows["is_residential"].any())
    
    

    def get_segment_coordinates(self, gdf_edges: gpd.GeoDataFrame, indexes : tuple[int, int], size_w : int, size_h : int, pixel_size : int = 1) -> tuple[tuple[float, float], tuple[float, float]]:
        min_x, min_y, max_x, max_y = gdf_edges.total_bounds
        segment_min_x = min_x + indexes[1] * size_w * pixel_size
        segment_max_x = segment_min_x + size_w * pixel_size
        if segment_max_x > max_x:
            segment_max_x = max_x
            
        segment_max_y = max_y - indexes[0] * size_h * pixel_size
        segment_min_y = segment_max_y - size_h * pixel_size
        if segment_min_y < min_y:
            segment_min_y = min_y

        return ((segment_min_x, segment_min_y), (segment_max_x, segment_max_y))
    

    def get_circle_radius(self, p1, p2, p3):
        # ze wzoru Herona i trójkąta wpisanego 
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        a = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        b = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
        c = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)
        
        if a < 0.0001 or b < 0.0001 or c < 0.0001:
            # print(f"a: {a}, b: {b}, c: {c}")
            return np.inf

        s = (a + b + c) / 2
        
        val = s * (s - a) * (s - b) * (s - c)
        if val <= 0:
            return np.inf
            
        area = np.sqrt(val)
        
        if area == 0:
            return np.inf
        
        R = (a * b * c) / (4 * area)
        return R
    

    def _get_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    

    def _get_deviation_angle(self, p_in, p_center, p_out):
        # Wektor wejściowy (u -> v)
        v1_x = p_center[0] - p_in[0]
        v1_y = p_center[1] - p_in[1]
        
        # Wektor wyjściowy (v -> w)
        v2_x = p_out[0] - p_center[0]
        v2_y = p_out[1] - p_center[1]
        

        angle1 = math.atan2(v1_y, v1_x)
        angle2 = math.atan2(v2_y, v2_x)
        
        diff = abs(angle1 - angle2)
        
        # do przedziału [0, PI]
        if diff > math.pi:
            diff = 2 * math.pi - diff
            
        return diff