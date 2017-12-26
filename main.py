from ui import test_ui
from sim import simulation
from sim import table
import random


def main():
    table_info = _get_table_info()
    sim = simulation.Simulation(table.TableInfo.from_dict(table_info))
    test_ui.run(sim, _get_inputs_function(sim))


def _get_inputs_function(sim: simulation.Simulation):
    time = 0
    rod_count = len(sim.table_info.rods) // 2

    def inputs_function(dt):
        nonlocal time
        time += dt
        return [(s, i) for s in [0, 1] for i in random_inputs(s)]

    def random_input(_, rod):
        return rod, random.random() * 1.0, random.random() * 2

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
    return {
        "length": 2.0,
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
                "x": 0.85,
                "foo_count": 5,
                "foo_spacing": 1/6
            },
            {
                "owner": 1,
                "x": 1 - 0.85,
                "foo_count": 5,
                "foo_spacing": 1 / 6
            },
            {
                "owner": 0,
                "x": 1 - 0.6,
                "foo_count": 3,
                "foo_spacing": 1 / 4
            },
            {
                "owner": 1,
                "x": 1 - 0.35,
                "foo_count": 2,
                "foo_spacing": 1 / 3
            },
            {
                "owner": 1,
                "x": 1 - 0.1,
                "foo_count": 1,
                "foo_spacing": 0
            }
        ],
        "foo": {
            "w": 0.02,
            "l": 0.08,
            "h": 0.09
        },
        "ball_r": 0.045,
        "goal_w": 0.25
    }


if __name__ == '__main__':
    main()
