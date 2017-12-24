from numpy import array, arange, argmax
from itertools import product
from sim.ai.NN import NN
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
        self.last_prediction = None
        self.last_state = None
        self.last_action_index = None
        self.lamda = 0.9  # TODO adjust
        self.alpha = 0.2  # TODO adjust
        self.epsilon = 0.25  # greedy policy
        # decreasing_rate will decrease epsilon such that in the future, when nn learned something
        # to not make anymore random choices
        self.__decreasing_rate = 0.992
        if load:
            self.__load(nn_file, actions_file)
            return

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
        self.actions = pickle.load(fd)
        fd.close()

    def save(self, nn_file: str="save.model", actions_file: str="save.actions"):
        self.model.save(nn_file)
        fd = open(actions_file, "wb")
        pickle.dump(self.actions, fd, protocol=0)  # protocol 0 for compatibility
        fd.close()

    def get_action(self, state):
        # TODO see if state need modification to be a vector with 1 dimension
        self.last_prediction = self.model.predict_action(state)
        self.last_state = state
        self.last_action_index = argmax(self.last_prediction)
        return self.actions[self.last_action_index]

    def get_action_off_policy(self, state):
        self.last_prediction = self.model.predict_action(state)
        self.last_state = state
        if random() < self.epsilon:  # should choose random an action
            self.last_action_index = randrange(0, len(self.actions))
            return self.actions[self.last_action_index]
        self.epsilon *= self.__decreasing_rate
        self.last_action_index = argmax(self.last_prediction)
        return self.actions[self.last_action_index]

    def update(self, action_based_reward: float, new_state):
        q_value = max(self.last_prediction)
        # TODO see if state need modification to be a vector with 1 dimension
        next_max_q_value = max(self.model.predict_action(new_state))
        q_value_updated = (1 - self.alpha) * q_value + \
                          self.alpha * (action_based_reward + next_max_q_value)
        # target will be last output of the network for state state, except for max value
        # which will be replaced by q_value_updated
        self.last_prediction[self.last_action_index] = q_value_updated
        self.model.update(self.last_state, self.last_prediction)
