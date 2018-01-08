from sim.simulation import Simulation
from numpy import array


class StateTemplate:

    def __init__(self, sim: Simulation):
        self.foosmans = []
        self.rev_foosmans = []

        table_length = sim.table_info.length

        for rod in sim.table_info.rods:
            _, x, foo_count, foo_dist, foo_offset = rod
            self.foosmans.append((x, foo_count, foo_dist))
            self.rev_foosmans.insert(0, (table_length - x, foo_count, foo_dist))
        self.max_position = complex(table_length, 1)

    @staticmethod
    def __complex_to_tuple(c: complex):
        return c.real, c.imag

    def get_states_from_sim(self, sim: Simulation):
        # TODO: this is not properly reversed.
        # Notably, rod angle and possibly offset are wrong.
        # Probably also ball position?
        # Also inputs!
        state_1 = list(StateTemplate.__complex_to_tuple(sim.state.ball[0])) +\
                  list(StateTemplate.__complex_to_tuple(sim.state.ball[1]))
        # Mirror this:
        # x = sim.table.length - x
        # y = 1 - x
        state_2 = list(StateTemplate.__complex_to_tuple(
             self.max_position - sim.state.ball[0])) +\
                  list(StateTemplate.__complex_to_tuple(-sim.state.ball[1]))

        # rods for player 1
        def unpack(p):
            return p[0][0], p[0][1], p[1][0], p[1][1]

        for i, ai_rod in zip(sim.state.rods, self.foosmans):
            state_1.extend(unpack(i))
            state_1.extend(ai_rod)

        # rods for player 2
        # both angles and offsets (incl velocities) must be reversed

        def neg_unpack(state_rod):
            (offset, offset_vel), (angle, angle_vel) = state_rod
            # offset is 0 - 1
            # angle is -1 - 1
            # velocities must be negated
            return 1 - offset, -offset_vel, -angle, -angle_vel

        for i, ai_rod in zip(reversed(sim.state.rods), self.rev_foosmans):
            state_2.extend(neg_unpack(i))
            state_2.extend(ai_rod)

        return state_1, state_2


class StateTemplatev2:

    def __init__(self, sim: Simulation):
        self.foosmans = []
        self.rev_foosmans = []

        table_length = sim.table_info.length

        for rod in sim.table_info.rods:
            _, x, foo_count, foo_dist, foo_offset = rod
            self.foosmans.append((x,
                                  self.get_foosmans_weight_center(foo_count, foo_dist)))
            self.rev_foosmans.insert(0, (table_length - x,
                                         self.get_foosmans_weight_center(foo_count,
                                                                         foo_dist)
                                         ))
        self.max_position = complex(table_length, 1)

    @staticmethod
    def __complex_to_tuple(c: complex):
        return c.real, c.imag

    def get_foosmans_weight_center(self, foo_count, foo_dist):
        if foo_dist == 0:
            return array([0.5])
        return array([(i + 1) * foo_dist for i in range(foo_count)])

    def apply_offset(self, player, offsets):
        # offsets must be in order from goal to middle
        rods = None
        if player == 0:
            rods = self.foosmans
        else:
            rods = self.rev_foosmans
        for offset, rod in zip(offsets, rods):
            if offset < 0:
                possible_offset = min(rod[1][0], abs(offset))
                rod[1] -= possible_offset
            else:
                possible_offset = min(1 - rod[1][-1], offset)
                rod[1] += possible_offset
