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

        # Normalizacja do długości 1
        norm = np.sqrt(A**2 + B**2)
        vx /= norm
        vy /= norm

        d = width / 2

        # Końce odcinka
        x1 = x_s + vx * d
        y1 = y_s + vy * d
        x2 = x_s - vx * d
        y2 = y_s - vy * d

        return (x1, y1), (x2, y2)

    def get_edge_polygon(self, edge):
        xs, ys = edge.geometry.xy
        xs = list(xs)
        ys = list(ys)
        right_side = []
        left_side = []
        
        if len(xs) == len(ys):
            for i in range(len(xs) - 1):
                x1 = xs[i]
                y1 = ys[i]
                x2 = xs[i + 1]
                y2 = ys[i + 1]
                coef = self.get_straight_line_coefficients(x1, y1, x2, y2)
                perpendicular_coef = self.get_perpendicular_line(coef[0], coef[1])
                (new_left_x1, new_left_y1), (new_right_x1, new_right_y1) = self.segment_from_line(perpendicular_coef[0], perpendicular_coef[1], x1, y1, edge.width_m)
                (new_left_x2, new_left_y2), (new_right_x2, new_right_y2)  = self.segment_from_line(perpendicular_coef[0], perpendicular_coef[1], x2, y2, edge.width_m)
                left_side.append((new_left_x1, new_left_y1))
                right_side.append((new_right_x1, new_right_y1))
                left_side.append((new_left_x2, new_left_y2))
                right_side.append((new_right_x2, new_right_y2))
        coords = left_side + right_side[::-1]
        coords.append(left_side[0])
        polygon = Polygon(coords)
        return polygon