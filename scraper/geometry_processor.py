import numpy as np
from shapely.geometry import Polygon

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
