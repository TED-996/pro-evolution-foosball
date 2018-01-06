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


def load_from_config():
    fd = open("config", "rt")
    global conf, pef_brain
    conf = json.load(fd)
    pef_brain = AI(load=False,
                   state_size=conf["state_size"],
                   rods_number=conf["rods_number"],
                   offset=conf["offset"],
                   angle_velocity=conf["angle_velocity"])  # see hidden layers field
    fd.close()


def load():
    global pef_brain
    pef_brain = AI(load=True)


def save():
    global pef_brain
    if pef_brain is not None:
        pef_brain.save()


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
    global state_template
    load_from_config()
    table_info = _get_table_info()
    sim = simulation.Simulation(table.TableInfo.from_dict(table_info))
    sim.on_goal.append(lambda side: print("Goal for {}".format(side)))
    state_template = StateTemplate(sim)  # see better place (bogdan)
    custom_ui.run(sim, _get_inputs_function(sim), _get_post_tick_function(sim))


last_input = None
action_taken = None

def _get_inputs_function(sim: simulation.Simulation):
    time = 0

    last_input = None
    last_time = time

    def inputs_function_nn(dt):  # this is a training function
        # has arg just to follow inputs_function signature
        global state_template
        nonlocal time
        nonlocal last_time
        nonlocal last_input
        global action_taken

        time += dt

        action_taken = False

        if last_input is None or time - last_time > random.random() + 0.5:
            action_taken = True
            state_1, state_2 = state_template.get_states_from_sim(sim)
            player_1 = pef_brain.get_action_off_policy(state_1, pef_brain.multiple_actions_off_policy)
            player_2 = pef_brain.get_action_off_policy(state_2, pef_brain.multiple_actions_off_policy)
            ret = list(map(lambda x: (0, x), player_1))
            ret.extend(map(lambda x: (1, x), player_2))
            last_input = ret
            last_time = time

        return last_input

    return inputs_function_nn


def _get_post_tick_function(sim: simulation.Simulation):
    def post_tick_function():
        global action_taken

        if action_taken:
            new_state_1, new_state_2 = state_template.get_states_from_sim(sim)
            reward_1 = sim.get_current_reward(0)
            reward_2 = sim.get_current_reward(1)
            pef_brain.update([reward_1, reward_2], [new_state_1, new_state_2])

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
