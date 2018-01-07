from ui import custom_ui
from sim import simulation
from sim import table
from ai.ai import AI
from ai.state_template import StateTemplate
import json
import random
import math
import sys


conf = {}
pef_brain : AI = None
state_template = None


def main():
    global state_template
    global pef_brain

    if "--load" in sys.argv:
        load()
    else:
        load_from_config()

    table_info = _get_table_info()
    sim = simulation.Simulation(table.TableInfo.from_dict(table_info))

    sim.on_goal.append(lambda side: print("Goal for {}".format(side)))
    sim.on_oob.append(lambda: print("WARNING: OOB, resetting"))

    sim.on_reset.append(lambda: pef_brain.flush_last_actions())

    state_template = StateTemplate(sim)  # see better place (bogdan)

    custom_ui.run(sim, _get_inputs_function(sim), _get_post_tick_function(sim), save)


def load_from_config():
    fd = open("config", "rt")
    global conf, pef_brain
    conf = json.load(fd)
    pef_brain = AI(load=False,
                   state_size=conf["state_size"],
                   rods_number=conf["rods_number"],
                   offset=conf["offset"],
                   angle_velocity=conf["angle_velocity"],
                   log_size=45)  # see hidden layers field
    fd.close()


def load():
    global pef_brain
    pef_brain = AI(load=True)


def save():
    print("saving...")
    global pef_brain
    if pef_brain is not None:
        pef_brain.save()


def get_actions(sim: simulation.Simulation):
    state_1, state_2 = state_template.get_states_from_sim(sim)
    return pef_brain.predict_action(state_1, pef_brain.multiple_actions), \
        pef_brain.predict_action(state_2, pef_brain.multiple_actions)


last_input = None
action_taken = None

def _get_inputs_function(sim: simulation.Simulation):
    time = 0

    next_time = 0

    def inputs_function_nn(dt):  # this is a training function
        # has arg just to follow inputs_function signature
        global state_template
        nonlocal time
        nonlocal next_time
        global last_input
        global action_taken

        time += dt

        action_taken = False

        if last_input is None or time >= next_time:
            action_taken = True
            state_1, state_2 = state_template.get_states_from_sim(sim)
            player_1 = pef_brain.get_action_off_policy(state_1, pef_brain.multiple_actions_off_policy)
            player_2 = pef_brain.get_action_off_policy(state_2, pef_brain.multiple_actions_off_policy)
            ret = list(map(lambda x: (0, x), player_1))
            ret.extend(map(lambda x: (1, x), player_2))
            last_input = ret
            next_time = time + 0.15
            # print("action taken")

        return last_input
        # return []

    return inputs_function_nn


def _get_post_tick_function(sim: simulation.Simulation):
    last_reward_1 = 0
    last_reward_2 = 0

    def post_tick_function():
        global action_taken
        nonlocal last_reward_1
        nonlocal last_reward_2

        if action_taken:
            new_state_1, new_state_2 = state_template.get_states_from_sim(sim)
            reward_1, penalty_1 = sim.get_current_reward(0)
            reward_2, penalty_2 = sim.get_current_reward(1)

            print(reward_1, penalty_1)

            pef_brain.update([reward_1 - last_reward_1 + penalty_1, reward_2 - last_reward_2 + penalty_2],
                             [new_state_1, new_state_2])

            last_reward_1 = reward_1
            last_reward_2 = reward_2

    def on_reset():
        nonlocal last_reward_1
        nonlocal last_reward_2

        last_reward_1 = 0
        last_reward_2 = 0

    sim.on_reset.append(on_reset)

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
                "foo_spacing": 1 / 2
            },
            {
                "owner": 1,
                "x": 0.6,
                "foo_count": 3,
                "foo_spacing": 1 / 3
            },
            {
                "owner": 0,
                "x": 0.9,
                "foo_count": 5,
                "foo_spacing": 1/5
            },
            {
                "owner": 1,
                "x": length - 0.9,
                "foo_count": 5,
                "foo_spacing": 1/5
            },
            {
                "owner": 0,
                "x": length - 0.6,
                "foo_count": 3,
                "foo_spacing": 1/3
            },
            {
                "owner": 1,
                "x": length - 0.35,
                "foo_count": 2,
                "foo_spacing": 1/2
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
