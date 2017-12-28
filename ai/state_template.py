from sim.simulation import Simulation


class StateTemplate:

    def __init__(self, sim: Simulation):
        self.foosmans = []
        self.rev_foosmans = []
        for rod in sim.table_info.rods:
            feature_selected = [rod[1], rod[2], rod[3]]
            self.foosmans.add(feature_selected)
            self.rev_foosmans.insert(0, feature_selected)

    @staticmethod
    def __complex_to_tuple(c: complex):
        return c.real, c.imag

    def get_states_from_sim(self, sim: Simulation):
        state_1 = []
        state_1.extend(StateTemplate.__complex_to_tuple(sim.state.ball[0]))
        state_1.extend(StateTemplate.__complex_to_tuple(sim.state.ball[1]))
        state_2 = state_1[:]

        # rods for player 1
        unpack = lambda p: [p[0][0], p[0][1], p[1][0], p[1][1]]
        for i, j in zip(sim.state.rods, self.foosmans):
            state_1.extend(unpack(i))
            state_1.extend(j)
        # rods for player 2
        for i, j in zip(reversed(sim.state.rods), self.rev_foosmans):
            state_2.extend(unpack(i))
            state_2.extend(j)

        return state_1, state_2
