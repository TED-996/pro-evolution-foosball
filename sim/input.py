from . import state


class Input:
    """
    Has:
        a side (0 or 1)
        a rod input, with:
            a rod index
            an offset velocity
            an angle velocity
    """

    def __init__(self, side, rod_idx, off_vel, ang_vel):
        self.side = side
        self.rod_idx = rod_idx
        self.off_vel = off_vel
        self.ang_vel = ang_vel