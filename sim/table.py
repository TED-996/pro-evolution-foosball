from . import state
import numpy as np
import pymunk


class TableInfo:
    """
    Has:
        A length (the maximum X coordinate, Y is clamped [0, 1]
        For each rod:
            its owner
            x coord (double)
            foosman count (int)
            distance between foosman centers (in [0, 1) )
             => side distance (lateral movement)
        Foosman width & height (double in [0, 1) )
        Goal width (in [0, 1) )
        Ball radius
    """

    def __init__(self, length, rods, foosman_size, goal_width, ball_radius):
        self.length = length
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
        self.ball_radius = ball_radius

    def get_init_state(self):
        return state.GameState(
            (0, 0),
            (np.array([0.5, 0.5]), np.array([0, 0])),
            [((0.5, 0), (0, 0)) for _ in self.rods]
        )

    def get_space(self):
        """
        Returns a pymunk Space with the ball and rods etc.
        and a dict of bodies in the form:
        {
            "ball": ball_body,
            "rods":[
                rod_0_bodies,
                rod_1_bodies,
                ...
                rod_n_bodies
            ]
        }

        (rod_x_bodies is a list of the bodies of foosmen)
        """
        space = pymunk.Space()

        ball_body = pymunk.Body(0, 0)

        rod_bodies = []
        for owner, x, foo_count, foo_dist, max_offset in self.rods:



