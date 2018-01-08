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
        self.foosman_length = sim.table_info.foosman_size[1]
        for rod in sim.table_info.rods:
            _, x, foo_count, foo_dist, foo_offset = rod
            self.foosmans.append([x,
                                  self.get_foosmans_positions(foo_count, foo_dist)])
            self.rev_foosmans.insert(0, [table_length - x,
                                         self.get_foosmans_positions(foo_count,
                                                                     foo_dist)
                                         ])
        self.max_position = complex(table_length, 1)

    @staticmethod
    def __complex_to_tuple(c: complex):
        return c.real, c.imag

    def get_foosmans_positions(self, foo_count, foo_dist):
        half_foosman_width = self.foosman_length / 2
        first_foo = (1 - foo_dist * (foo_count - 1)) / 2
        weight_centers = [first_foo + i * foo_dist for i in range(foo_count)]
        return array([i + j * half_foosman_width
                      for i in weight_centers
                      for j in [-1, 1]])

    @staticmethod
    def apply_offset(offsets, rods):
        # offsets must be in order from X = 0 to X = max_X
        assert len(offsets) == len(rods), \
            "Offsets must have length {}".format(len(rods))
        for offset, rod in zip(offsets, rods):
            if offset < 0:
                possible_offset = min(rod[1][0], abs(offset))
                rod[1] -= possible_offset
            else:
                possible_offset = min(1 - rod[1][-1], offset)
                rod[1] += possible_offset

    def reset(self):
        for foosman in self.foosmans:
            # check side distance
            # must be 0 if rods are in initial position
            offset = (1 - foosman[1][-1] - foosman[1][0]) / 2
            foosman[1] += offset
        for rev_foosman in self.rev_foosmans:
            # check side distance
            # must be 0 if rods are in initial position
            offset = (1 - rev_foosman[1][-1] - rev_foosman[1][0]) / 2
            rev_foosman[1] += offset

    def get_states_from_sim(self, sim: Simulation):
        # TODO: this is not properly reversed.
        # Notably, rod angle and possibly offset are wrong.
        # Probably also ball position?
        # Also inputs!
        state_1 = list(StateTemplatev2.__complex_to_tuple(sim.state.ball[0])) +\
                  list(StateTemplatev2.__complex_to_tuple(sim.state.ball[1]))
        # Mirror this:
        # x = sim.table.length - x
        # y = 1 - x
        state_2 = list(StateTemplatev2.__complex_to_tuple(
             self.max_position - sim.state.ball[0])) +\
                  list(StateTemplatev2.__complex_to_tuple(-sim.state.ball[1]))

        # rods for player 1
        def unpack(p):
            return p[0][0], p[0][1], p[1][0], p[1][1]

        self.apply_offset([rod[0][0] for rod in sim.state.rods],
                          self.foosmans)

        for i, ai_rod in zip(sim.state.rods, self.foosmans):
            state_1.append(ai_rod[0])
            state_1.extend(ai_rod[1])
            state_1.extend(unpack(i)[1:])

        # rods for player 2
        # both angles and offsets (incl velocities) must be reversed
        self.apply_offset([1 - rod[0][0] for rod in reversed(sim.state.rods)],
                          self.rev_foosmans)

        def neg_unpack(state_rod):
            (offset, offset_vel), (angle, angle_vel) = state_rod
            # offset is 0 - 1
            # angle is -1 - 1
            # velocities must be negated
            return 1 - offset, -offset_vel, -angle, -angle_vel

        for i, ai_rod in zip(reversed(sim.state.rods), self.rev_foosmans):
            state_2.append(ai_rod[0])
            state_2.extend(ai_rod[1])
            state_2.extend(neg_unpack(i)[1:])

        return state_1, state_2