from ui import custom_ui
from sim import simulation
from sim import table
from ai.ai import AI
import random

# TODO configure model from a config file
state_size = 10  # to be adjusted
rods_number = 10  # to be adjusted
offset = [0.2, 0.1, 0.0, -0.1, -0.2]  # to be adjusted
angle_velocity = [0.75, 0.25, 0.0, -0.25, -0.75]  # to be adjusted
pef_brain = AI(load=False,
               state_size=state_size,
               rods_number=rods_number,
               offset=offset,
               angle_velocity=angle_velocity)  # see hidden layers field


def main():
    table_info = _get_table_info()
    sim = simulation.Simulation(table.TableInfo.from_dict(table_info))
    custom_ui.run(sim, _get_inputs_function(sim))


def complex_to_tuple(c):
    return c.real, c.imag


def get_states_from_sim(sim: simulation.Simulation):
    state_1 = []
    state_1.extend(complex_to_tuple(sim.state.ball[0]))
    state_1.extend(complex_to_tuple(sim.state.ball[1]))
    state_2 = state_1[:]

    # rods for player 1
    unpack = lambda p: [p[0][0], p[0][1], p[1][0], p[1][1]]
    for i, j in zip(sim.state.rods, sim.table_info.rods):
        state_1.extend(unpack(i))
    for i in reversed(sim.state.rods):
        state_2.extend(unpack(i))
    return state_1, state_2

def predict_actions(sim: simulation.Simulation):
    """
    :return: return 2 list with action for each rod of each player
    """
    s1, s2 = get_states_from_sim(sim)
    p1_actions = pef_brain.get_action(s1, pef_brain.multiple_actions)
    p2_actions = pef_brain.get_action(s2, pef_brain.multiple_actions)
    return p1_actions, p2_actions


def _get_inputs_function(sim: simulation.Simulation):
    time = 0
    rod_count = len(sim.table_info.rods) // 2

    def inputs_function(dt):
        nonlocal time
        time += dt
        return [(s, i) for s in [0, 1] for i in random_inputs(s)]

    def random_input(_, rod):
        return rod, (random.random() - 0.5) * 1.0, (random.random() - 0.5) * 16.0

    last_input = ([random_input(0, rod) for rod in range(rod_count)],
                  [random_input(1, rod) for rod in range(rod_count)])
    last_time = time

    def random_inputs(side):
        nonlocal last_time
        nonlocal last_input

        input = last_input[side]
        if time - last_time > random.random() + 0.1:
            rod, offset, angle = input.pop(0)
            input.append(random_input(side, rod))
            last_time = time

        return last_input[side]

    return inputs_function


def _get_table_info():
    length = 2.0
    return {
        "length": length,
        "rods": [
            {
                "owner": 0,
                "x": 0.1,
                "foo_count": 1,
                "foo_spacing": 0
            },
            {
                "owner": 0,
                "x": 0.35,
                "foo_count": 2,
                "foo_spacing": 1/3
            },
            {
                "owner": 1,
                "x": 0.6,
                "foo_count": 3,
                "foo_spacing": 1/4
            },
            {
                "owner": 0,
                "x": 0.9,
                "foo_count": 5,
                "foo_spacing": 1/6
            },
            {
                "owner": 1,
                "x": length - 0.9,
                "foo_count": 5,
                "foo_spacing": 1 / 6
            },
            {
                "owner": 0,
                "x": length - 0.6,
                "foo_count": 3,
                "foo_spacing": 1 / 4
            },
            {
                "owner": 1,
                "x": length - 0.35,
                "foo_count": 2,
                "foo_spacing": 1 / 3
            },
            {
                "owner": 1,
                "x": length - 0.1,
                "foo_count": 1,
                "foo_spacing": 0
            }
        ],
        "foo": {
            "w": 0.04,
            "l": 0.08,
            "h": 0.09
        },
        "ball_r": 0.045,
        "goal_w": 0.25
    }


if __name__ == '__main__':
    main()
