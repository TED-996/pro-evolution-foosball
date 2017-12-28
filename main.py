from ui import custom_ui
from sim import simulation
from sim import table
from ai.ai import AI
from ai.state_template import StateTemplate
import json
import random


conf = {}
pef_brain = None
state_template = None


def load():
    fd = open("config", "rt")
    global conf, pef_brain
    conf = json.load(fd)
    pef_brain = AI(load=False,
                   state_size=conf["state_size"],
                   rods_number=conf["rods_number"],
                   offset=conf["offset"],
                   angle_velocity=conf["angle_velocity"])  # see hidden layers field
    fd.close()


def get_actions(sim: simulation.Simulation):
    state_1, state_2 = state_template.get_states_from_sim(sim)
    return pef_brain.predict_action(state_1, pef_brain.multiple_actions), \
        pef_brain.predict_action(state_2, pef_brain.multiple_actions)


def act_and_update_template(sim: simulation.Simulation):
    # THIS FUNCTION IS JUST A TEMPLATE FOR HOW TO INTERACT WITH AI
    # while loop
    state_1, state_2 = state_template.get_states_from_sim(sim)
    player_1 = pef_brain.get_action_off_policy(state_1, pef_brain.multiple_actions_off_policy)
    player_2 = pef_brain.get_action_off_policy(state_2, pef_brain.multiple_actions_off_policy)

    # apply input (player_1 and player_2) to sim
    # update gui

    new_state_1, new_state_2 = state_template.get_states_from_sim(sim)
    reward_1 = sim.get_current_reward(0)
    reward_2 = sim.get_current_reward(1)
    pef_brain.update([reward_1, reward_2], [new_state_1, new_state_2])
    # end loop


def main():
    table_info = _get_table_info()
    sim = simulation.Simulation(table.TableInfo.from_dict(table_info))
    sim.on_goal.append(lambda side: print("Goal for {}".format(side)))
    state_template = StateTemplate(sim)  # see better place (bogdan)
    custom_ui.run(sim, _get_inputs_function(sim), _get_post_tick_function(sim))


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


def _get_post_tick_function(sim: simulation.Simulation):
    def post_tick_function():
        pass

    return post_tick_function

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
