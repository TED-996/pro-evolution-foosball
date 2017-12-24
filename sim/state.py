class GameState:
    """
    Has:
        A score (tuple of 2 ints)
        The coordinates of the ball (complex, in [0, 1] ) and its velocities
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
            ball ([complex, complex]): The position of the ball (in [0,1]) and its velocity (in units/sec)
            rods (list of ((double, double), (double, double))): for each rod left to right (home to away),
                (its offset in [0,1] and its velocity)
                and
                (its angle in [-1,1]) and its velocity)
        """
        self.score : (int, int) = score
        self.ball : (complex, complex) = ball
        self.rods : [((float, float), (float, float))] = rods

    def clone(self):
        return GameState(self.score, self.ball, self.rods[:])

    def apply_inputs(self, inputs):

