import numpy as np


class GameState:
    """
    Has:
        A score (tuple of 2 ints)
        The coordinates of the ball (np.array, in [0, 1] ) and its velocities
        A list of rods, each with:
            An offset in [0, 1], the sideways coord
            An offset velocity in units/sec
            An angle in [-1, 1] where -1 is -90deg and 1 is 90deg
            An angle velocity in units/sec
    """

    def __init__(self, score, ball, rods):
        """
        Initialize a new PefState

        Args:
            score (int, int): The score of the match
            ball ([np.array[2], np.array[2]]): The position of the ball (in [0,1]) and its velocity (in units/sec)
            rods (list of ((double, double), (double, double))): for each rod left to right (home to away),
                (its offset in [0,1] and its velocity)
                and
                (its angle in [-1,1]) and its velocity)
        """
        self.score = score
        self.ball = ball
        self.rods = rods

    def clone(self):
        return GameState(self.score, [np.copy(self.ball[0]), np.copy(self.ball[1])], self.rods[:])

    def apply_inputs(self, inputs):
        