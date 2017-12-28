from numpy import array, arange, argmax
from itertools import product
from ai.NN import NN
import pickle
from random import random, randrange


class AI:

    def __init__(self, load: bool = False,
                 state_size: int = 0,
                 rods_number: int = 0,
                 offset=None,
                 angle_velocity=None,
                 hidden_layers=(100, 100),
                 nn_file: str = "save.model",
                 actions_file: str = "save.actions"):
        """
        :param load: specify if model should be load
        :param state_size: how many attributes has a state
        :param rods_number: number of rods of a player
        :param offset: represents a vector with scalars that will move rod to left/right
        :param angle_velocity: represents a vector with rates of change of a rodsman angle
        :param hidden_layers: a vector of values that represents how many units have each layer
        :param nn_file: file to save neural network
        :param actions_file: file to save actions_file
        """
        self.actions = None
        self.model = None
        self.last_predictions = []
        self.last_states = []
        self.last_actions_index = []
        self.lamda = 0.9  # TODO adjust
        self.alpha = 0.2  # TODO adjust
        self.epsilon = 0.25  # greedy policy
        # decreasing_rate will decrease epsilon such that in the future, when nn learned something
        # to not make anymore random choices
        self.__decreasing_rate = 0.992
        if load:
            self.__load(nn_file, actions_file)
            return

        self.rods_number = rods_number
        self.actions = array(
            [action for action in product(arange(rods_number),
                                          offset,
                                          angle_velocity)]
        )
        self.model = NN(input_dim=state_size,
                        hidden_layers=hidden_layers,
                        output_dim=len(self.actions))
        self.model.compile()

    def __load(self, nn_file, actions_file):
        self.model = NN(load_file=nn_file)
        fd = open(actions_file, "rb")
        self.rods_number, self.actions = pickle.load(fd)
        fd.close()

    def save(self, nn_file: str="save.model", actions_file: str="save.actions"):
        self.model.save(nn_file)
        fd = open(actions_file, "wb")
        to_save = (self.rods_number, self.actions)
        pickle.dump(to_save, fd, protocol=0)  # protocol 0 for compatibility
        fd.close()

    def __compute_and_backup(self, state):
        self.last_predictions.append(self.model.predict_action(state))
        self.last_states.append(state)

    def one_action(self, q_values):
        return [argmax(q_values)]

    def multiple_actions(self, q_values):
        actions_idxs = []
        slice_size = len(self.actions) / self.rods_number
        for i in range(self.rods_number):
            actions_idxs.append(i * slice_size + argmax(q_values[i * slice_size:(i + 1) * slice_size]))
        return actions_idxs

    def get_action(self, state, action_selector):
        """
        :param state: a state of the current game
        :param action_selector: may be one of the following functions: ane_action, multiple_actions
        :return:
        """
        # TODO see if state need modification to be a vector with 1 dimension
        self.__compute_and_backup(state)
        self.last_actions_index.append(action_selector(self.last_predictions))
        return [self.actions[i] for i in self.last_actions_index]

    def one_action_off_policy(self, rand, q_values):
        if rand:
            return [randrange(0, len(self.actions))]
        else:
            return self.one_action(q_values)

    def multiple_actions_off_policy(self, rand, q_values):
        slice_size = len(self.actions) / self.rods_number
        if rand:
            return [i * slice_size + randrange(0, len(self.actions))
                    for i
                    in range(self.rods_number)]
        else:
            return self.multiple_actions(q_values)

    def get_action_off_policy(self, state, action_selector):
        self.__compute_and_backup(state)
        if random() < self.epsilon:  # should choose an action random
            self.last_actions_index.append(action_selector(True, None))
            return [self.actions[i] for i in self.last_actions_index]
        self.epsilon *= self.__decreasing_rate
        self.last_actions_index.append(action_selector(False, self.last_predictions))
        return [self.actions[i] for i in self.last_actions_index]

    def update(self, action_based_reward: float, new_states):
        assert len(action_based_reward) == len(new_states), "must have reward for each new_state"
        assert len(new_states) == len(self.last_actions_index), \
            "must have the same amount of new_states as {}".format(len(self.last_actions_index))

        q_values = [[self.last_predictions[i][j]
                    for j in self.last_actions_index[i]]
                    for i in len(self.last_actions_index)]

        action_selector = self.one_action if len(q_values[0]) == 1 else self.multiple_actions

        for i in len(new_states):
            next_max_q_values = action_selector(self.model.predict_action(new_states[i]))

            q_values_updated = [(1 - self.alpha) * q + self.alpha * (action_based_reward[i] + next_q)
                                for q, next_q
                                in zip(q_values[i], next_max_q_values)]

            for j, update in zip(self.last_actions_index[i], q_values_updated):
                self.last_predictions[i][j] = update
        self.model.update(self.last_states, self.last_predictions)
        self.last_states.clear()
        self.last_actions_index.clear()
        self.last_predictions.clear()

    def predict_action(self, state, action_selector):
        actions_idxs = action_selector(self.model.predict_action(state))
        return [self.actions[i] for i in actions_idxs]