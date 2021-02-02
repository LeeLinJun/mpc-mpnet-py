from enum import Enum

vis_eps = 1e-10


class Dynamics(Enum):
    Acrobot = 1
    Cartpole = 2
    Car = 3
    Quadorotor = 4


def line_line_cc(x1, y1, x2, y2, x3, y3, x4, y4):
    uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / \
        ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1) + vis_eps)
    uB = ((x2 - x1) * (y1 - y3) - (y2-y1) * (x1 - x3)) / \
        ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1) + vis_eps)
    if uA >= 0. and uA <= 1. and uB >= 0. and uB <= 1.:
        # intersection
        return True
    # collision free
    return False
