from . import state


class Input:
    """
    Has:
        a side (0 or 1)
        0, 1 or 2 rod inputs, for each:
            an index
            an offset acceleration
            an offset angle
    """

    def __init__(self, side, rods):
        self.side = side
        self.rods = []
        for idx, t_offset, t_angle in rods:
            self.rods.append((idx, t_offset, t_angle))

        self._check_valid()

    def _check_valid(self):
        assert self.side == 0 or self.side == 1
        assert len(self.rods) <= 2

        # No duplicate rods
        idxs = {r[0] for r in self.rods}
        assert len(idxs) == len(self.rods)
