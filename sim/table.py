from . import state
import numpy as np


class TableInfo:
    """
    Has:
        Aspect ratio r = (x / y); y = x / r
            x is the wide side, y is the narrow side
        For each rod:
            its owner
            x coord (double)
            foosman count (int)
            distance between foosman centers (in [0, 1) )
             => side distance (lateral movement)
        Foosman width & height (double in [0, 1) )
        Goal width (in [0, 1) )
    """

    def __init__(self, ratio, rods, foosman_size, goal_width):
        self.ratio = ratio
        self.rods = []
        for owner, x, foo_count, foo_dist in rods:
            self.rods.append((
                owner,
                x,
                foo_count,
                foo_dist,
                1 - foo_count * foo_dist
            ))
        self.foosman_size = foosman_size
        self.goal_width = goal_width

    def get_init_state(self):
        return state.GameState(
            (0, 0),
            (np.array([0.5, 0.5]), np.array([0, 0])),
            [((0.5, 0), (0, 0)) for _ in self.rods]
        )
