from sim.simulation import Simulation
from ai.ai import AI
from ai.state_template import StateTemplate


class Manager:
    """
    A class that will handle flow of states to the network
    TODO: toggle to switch between update, non-update
    """
    def __init__(self, brain: AI,
                 sim: Simulation,
                 state_template: StateTemplate):
        self.brain = brain
        self.sim = sim
        self.state_template = state_template
        # will preserve (reward, state) until it will be passed to AI update
        self.backup = None

    def _make_action(self, side: int):
        # player on the side 'side' make a move
        # based on actions taken, update sim
        # return a pair of state, reward
        state = self.state_template.get_state_from_sim_for_player(side, self.sim)
        reward = self.sim.get_current_reward(side)
        for input in self.brain.get_action_off_policy(state, self.brain.multiple_actions_off_policy):
            self.sim.apply_inputs(side, input)
        reward -= self.sim.get_current_reward(side)
        return self.state_template.get_state_from_sim_for_player(side, self.sim), reward

    def make_action_for_both_players(self):
        new_states = [None, None]
        rewards = [None, None]
        new_states[0], rewards[0] = self._make_action(0)
        new_states[1], rewards[1] = self._make_action(1)
        self.backup = (rewards, new_states)
        return rewards, new_states

    def update(self):
        if self.backup is not None:
            self.brain.update(*self.backup)
            del self.backup

