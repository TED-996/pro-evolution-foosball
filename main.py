from ui import test_ui
from sim import simulation
from sim import table


def main():
    table_info = _get_table_info()
    sim = simulation.Simulation(table.TableInfo.from_dict(table_info))
    test_ui.run(sim, _get_inputs_function(sim))



def _get_inputs_function(sim:simulation.Simulation):
    time = 0
    def inputs_function(dt):
        global time
        time += dt
        return [(s, i) for s in [0, 1] for i in random_input(s)]

    def random_input(side):
        pass

    return inputs_function


def _get_table_info():
    return {
        "length": 2.0,
        "rods": [
            {
                "owner": 0,
                "x": 0.1,
                "foo_count": 1,
                "foo_spacing": 1
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
                "foo_spacing": 1
            }
        ],
        "foo": {
            "w": 0.02,
            "l": 0.08,
            "h": 0.09
        },
        "ball_r": 0.045
    }