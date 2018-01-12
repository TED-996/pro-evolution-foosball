from sim.simulation import Simulation
from ai.ai import AI
from ai.state_template import StateTemplate


class Manager:
    """
    A class that will handle flow of states to the network
    """
    def __init__(self, brain: AI,
                 sim: Simulation,
                 state_template: StateTemplate):
        self.brain = brain
        self.sim = sim
        self.state_template = state_template
        # will preserve (reward, state) until it will be passed to AI update
        self.backup = None
        self.on_make_action = self._make_action_for_both_players
        self.on_update = self._update

    def toggle(self, active):
        # if active is true then made updates on network
        # else, simple predict actions
        if active:
            self.on_make_action = self._make_action_for_both_players
            self.on_update = self._update
        else:
            self.on_make_action = self._make_actions
            self.on_update = lambda: None

    def _get_single_player_action(self, side: int):
        # player on the side 'side' make a move
        # based on actions taken, update sim
        # return a pair of state, reward
        state = self.state_template.get_state_from_sim_for_player(side, self.sim)
        reward = self.sim.get_current_reward(side)
        for input in self.brain.get_action_off_policy(state, self.brain.multiple_actions_off_policy):
            self.sim.apply_inputs(side, input)
        reward -= self.sim.get_current_reward(side)
        return self.state_template.get_state_from_sim_for_player(side, self.sim), reward

    def _make_action_for_both_players(self):
        # this version is for update
        new_states = [None, None]
        rewards = [None, None]
        new_states[0], rewards[0] = self._get_single_player_action(0)
        new_states[1], rewards[1] = self._get_single_player_action(1)
        self.backup = (rewards, new_states)

    def _make_actions(self):
        # this version is for version without update
        # we have a state - give it to the first player and apply his actions
        state = self.state_template.get_state_from_sim_for_player(0, self.sim)
        for input in self.brain.predict_action(state, self.brain.multiple_actions):
            self.sim.apply_inputs(0, input)
        # after first player made his action, now, new state pass to the second player
        # and apply his actions
        state = self.state_template.get_state_from_sim_for_player(1, self.sim)
        for input in self.brain.predict_action(state, self.brain.multiple_actions):
            self.sim.apply_inputs(1, input)

    def _update(self):
        if self.backup is not None:
            self.brain.update(*self.backup)
            del self.backup

    def make_action(self):
        # interface for make action (toggle will switch between semantic of this function)
        self.on_make_action()

    def update(self):
        # interface for update (toggle will switch between semantic of this function)
        self.on_update()

