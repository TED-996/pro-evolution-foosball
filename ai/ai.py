from numpy import array, arange, argmax
from numpy.random import choice
from itertools import product
from ai.NN import NN
import pickle
from random import random, randrange, randint
from collections import deque
from math import floor


class AI:
    MEMORY_DIMENSION = 90000

    def __init__(self, load: bool = False,
                 state_size: int = 0,
                 rods_number: int = 0,
                 offset=None,
                 angle_velocity=None,
                 hidden_layers=(100, 100),
                 log_size=10,
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
        self.last_predictions = deque(maxlen=2 * log_size)
        self.last_states = deque(maxlen=2 * log_size)
        self.last_actions_index = deque(maxlen=2 * log_size)
        self.last_reward_sums = deque([0, 0], maxlen=2 * log_size + 2)
        self.log_size = log_size
        self.lamda = 0.6
        self.alpha = 0.9
        self.epsilon = 0.5  # greedy policy
        self.__epsilon_backup = self.epsilon
        # decreasing_rate will decrease epsilon such that in the future, when nn learned something
        # to not make anymore random choices
        self.__decreasing_rate = 0.99997

        # memory replay
        self.memory_state = deque(maxlen=AI.MEMORY_DIMENSION)
        self.memory_target = deque(maxlen=AI.MEMORY_DIMENSION)
        # with save_probability save a memory with consist of a state and a target
        self.save_probability = 0.3
        if load:
            self.__load(nn_file, actions_file)
            return

        self.rods_number = rods_number
        self.actions = array(
            [action for action in product(arange(rods_number),
                                          offset,
                                          angle_velocity)]
        )
        self.batch_size = int(floor((2 * log_size) ** 0.5))
        self.model = NN(input_dim=state_size,
                        hidden_layers=hidden_layers,
                        output_dim=len(self.actions),
                        batch_size=self.batch_size)

        self.model.compile()

    def __load(self, nn_file, actions_file):
        self.model = NN(load_file=nn_file)
        fd = open(actions_file, "rb")
        self.rods_number, self.actions, self.epsilon, self.lamda, self.batch_size = pickle.load(fd)
        self.model.batch_size = self.batch_size
        fd.close()

    def save(self, nn_file: str="save.model", actions_file: str="save.actions"):
        self.model.save(nn_file)
        print("saving ai...")
        fd = open(actions_file, "wb")
        to_save = (self.rods_number, self.actions, self.epsilon, self.lamda, self.batch_size)
        pickle.dump(to_save, fd, protocol=0)  # protocol 0 for compatibility
        fd.close()

    def __compute_and_backup(self, state):
        self.last_predictions.append(self.model.predict_action(state))
        self.last_states.append(state)

    # noinspection PyMethodMayBeStatic
    def one_action(self, q_values):
        return [argmax(q_values)]

    def multiple_actions(self, q_values):
        actions_idxs = []
        slice_size = int(len(self.actions) // self.rods_number)
        for i in range(self.rods_number):
            actions_idxs.append(i * slice_size + argmax(q_values[i * slice_size:(i + 1) * slice_size]))
        return actions_idxs

    def get_action(self, state, action_selector):
        """
        :param state: a state of the current game
        :param action_selector: may be one of the following functions: ane_action, multiple_actions
        :return:
        """
        self.__compute_and_backup(state)
        self.last_actions_index.append(action_selector(self.last_predictions[-1]))
        return [self.actions[i] for i in self.last_actions_index[-1]]

    def one_action_off_policy(self, rand, q_values):
        if rand:
            return [randrange(0, len(self.actions))]
        else:
            return self.one_action(q_values)

    def multiple_actions_off_policy(self, rand, q_values):
        slice_size = int(len(self.actions) // self.rods_number)
        if rand:
            return [i * slice_size + randrange(0, slice_size)
                    for i
                    in range(self.rods_number)]
        else:
            return self.multiple_actions(q_values)

    def get_action_off_policy(self, state, action_selector):
        self.__compute_and_backup(state)

        if random() < self.epsilon:  # should choose an action random
            self.epsilon *= self.__decreasing_rate
            self.last_actions_index.append(action_selector(True, None))
            return [self.actions[i] for i in self.last_actions_index[-1]]

        self.last_actions_index.append(action_selector(False, self.last_predictions[-1]))
        return [self.actions[i] for i in self.last_actions_index[-1]]

    def update(self, action_based_reward, new_states):
        assert len(action_based_reward) == len(new_states), "must have reward for each new_state"
        assert len(action_based_reward) == 2, "exactly 2 players supported ATM"

        for idx in range(0, len(self.last_reward_sums), 2):
            self.last_reward_sums[idx] += action_based_reward[0]
            self.last_reward_sums[idx + 1] += action_based_reward[1]

        self.last_reward_sums.extend(action_based_reward)

        q_values = [[self.last_predictions[i][j]
                    for j in self.last_actions_index[i]]
                    for i in range(len(self.last_actions_index))]
        action_selector = self.one_action if len(q_values[0]) == 1 else self.multiple_actions

        for i in range(len(q_values) - 2):
            next_q_values = q_values[i + 2]
            q_values_updated = [
                # TODO: not sure about self.last_rewards[i]
                (1 - self.alpha) * q + self.alpha * (self.last_reward_sums[i] / ((len(q_values) + 1 - i) // 2)
                    + (self.lamda * next_q))
               for q, next_q
               in zip(q_values[i], next_q_values)]

            for j, update in zip(self.last_actions_index[i], q_values_updated):
                self.last_predictions[i][j] = update

        for i in range(len(new_states)):
            next_max_q_values = action_selector(self.model.predict_action(new_states[i]))

            q_values_updated = [
                (1 - self.alpha) * q + self.alpha * (action_based_reward[i] + self.lamda * next_q)
                for q, next_q
                in zip(q_values[-2 + i], next_max_q_values)]

            for j, update in zip(self.last_actions_index[-2 + i], q_values_updated):
                self.last_predictions[-2 + i][j] = update

        if random() < self.save_probability:
            self.memory_state.append(self.last_states[0])
            self.memory_target.appendleft(self.last_predictions[0])

        if random() <= 0.5:
            self.model.update(self.last_states, self.last_predictions)
        else:
            self.from_memory_update()
        # we trust more in next move when network learn more
        self.lamda += self.lamda * 1.e-7
        if self.lamda > 1:
            print("lambda = {}".format(self.lamda))
            self.lamda = 1

    def predict_action(self, state, action_selector):
        actions_idxs = action_selector(self.model.predict_action(state))
        return [self.actions[i] for i in actions_idxs]

    def flush_last_actions(self):
        self.last_states.clear()
        self.last_actions_index.clear()
        self.last_predictions.clear()

    def switch_random_action(self, activate):
        if not activate:
            self.__epsilon_backup = self.epsilon
            self.epsilon = 0
        else:
            self.epsilon = self.__epsilon_backup

    def from_memory_update(self):
        if len(self.memory_state) < 1000:
            return
        idxs = arange(len(self.memory_state))
        size = randint(200, 1000)
        sample = choice(idxs, size, replace=False)
        self.model.update(array([self.memory_state[i] for i in sample]),
                          array([self.memory_target[i] for i in sample]),
                          False, 3)
