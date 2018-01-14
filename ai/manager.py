from sim.simulation import Simulation
from ai.ai import AI
from ai.state_template import StateTemplate
from numpy import array


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

    def make_action(self, _):
        # interface for make action (toggle will switch between semantic of this function)
        self.on_make_action()

    def update(self):
        # interface for update (toggle will switch between semantic of this function)
        self.on_update()

    def switch_random_action(self, toggle):
        self.brain.switch_random_action(toggle)

    def save(self):
        self.brain.save()


class IndependentManager:

    def __init__(self, brain_1: AI, brain_2: AI, sim: Simulation, state_template: StateTemplate):
        self.brain_1 = brain_1
        self.brain_2 = brain_2
        self.sim = sim
        self.state_template = state_template
        # will preserve (reward, state) until it will be passed to AI update
        self.backup_1 = None
        self.backup_2 = None
        self.on_make_action = self._make_action_for_update
        self.on_update = self._update

    def toggle(self, active):
        # if active is true, the perform and update
        if active:
            self.on_make_action = self._make_action_for_update
            self.on_update = self._update
        else:
            self.on_make_action = self._make_action
            self.on_update = lambda: None

    def _make_one_action_for_update(self):
        state1, state2 = self.state_template.get_states_from_sim(self.sim)
        player1 = self.brain_1.get_action_off_policy(state1, self.brain_1.multiple_actions_off_policy)
        player2 = self.brain_2.get_action_off_policy(state2, self.brain_2.multiple_actions_off_policy)
        score = array([self.sim.get_current_reward(0), self.sim.get_current_reward(1)])
        for input in player1:
            self.sim.apply_inputs(0, input)
        for input in player2:
            self.sim.apply_inputs(1, input)
        score = score - array([self.sim.get_current_reward(0), self.sim.get_current_reward(1)])
        return score, self.state_template.get_states_from_sim(self.sim)

    def _make_action_for_update(self, tick):
        score_1, new_state_1 = self._make_one_action_for_update()
        self.sim.tick(tick)
        score_2, new_state_2 = self._make_one_action_for_update()
        self.backup_1 = (score_1[0], score_2[0]), (new_state_1[0], new_state_2[0])
        self.backup_2 = (score_1[1], score_2[1]), (new_state_1[1], new_state_2[1])

    def _update(self):
        if self.backup_1 is not None and \
           self.backup_2 is not None:
            self.brain_1.update(*self.backup_1)
            self.brain_2.update(*self.backup_1)
            del self.backup_1
            del self.backup_2

    def _make_action(self, _):
        state1, state2 = self.state_template.get_states_from_sim(self.sim)
        player1 = self.brain_1.predict_action(state1, self.brain_1.multiple_actions)
        player2 = self.brain_2.predict_action(state2, self.brain_2.multiple_actions)
        for input in player1:
            self.sim.apply_inputs(0, input)
        for input in player2:
            self.sim.apply_inputs(1, input)

    def save(self):
        self.brain_1.save(actions_file="save.actions1", nn_file="save.model1")
        self.brain_2.save(actions_file="save.actions2", nn_file="save.model2")

    def make_action(self, tick):
        # interface for make action (toggle will switch between semantic of this function)
        self.on_make_action(tick)

    def update(self):
        # interface for update (toggle will switch between semantic of this function)
        self.on_update()

    def switch_random_action(self, toggle):
        self.brain_1.switch_random_action(toggle)
        self.brain_2.switch_random_action(toggle)
