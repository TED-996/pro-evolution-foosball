from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
import itertools
from numpy import arange, array, argmax


class NN:

    def __init__(self, input_dim, hidden_layers, rods_number, offset, angle_velocity):
        """
        :param hidden_layers: number of units on each hidden layer
        :param input_dim: input layer dimension(number of units)
        :param rods_number: represents number of rods that a player have
        :param offset: represents a vector with scalars that will move rod to left/right
        :param angle_velocity: represents a vector with rates of change of a rodsman angle
        """
        assert len(hidden_layers) > 0, "NN must contain al least one hidden layer"
        assert len(offset) > 0, "offset vector must have at least len 1"
        assert len(angle_velocity) > 0, "angle_velocity vector must have at lest len 1"
        # compute cartesian product of the form (rod_index, offset, angle_velocity)
        self.actions = array(
            [action for action in itertools.product(arange(rods_number),
                                                    offset,
                                                    angle_velocity)]
        )
        # add first and last layer in network
        nn_layers = [Dense(units=hidden_layers[0],
                           activation="sigmoid",
                           input_dim=input_dim),
                     Dense(units=len(self.actions),
                           activation="relu")]
        # if there are more hidden layers, add them before last layer
        for layer_size in range(1, len(hidden_layers)):
            nn_layers.insert(-1, Dense(units=layer_size,
                                       activation="sigmoid"))
        self.model = Sequential(nn_layers)
        self.target = None
        self.last_input = None
        self.last_action = None

    def compile(self, lr=0.1, rho=0.9, epsilon=1e-08, decay=0.0):
        # TODO improvement of hyper-parameters of NN
        self.model.compile(optimizer=RMSprop(lr=lr, rho=rho, epsilon=epsilon, decay=decay),
                           loss="categorical_crossentropy",
                           metrics=["accuracy"])

    def predict_action(self, state):
        """
        :param state: must be a vector with all values that will represent a state
        :return: a tuple of action, that is considered to be the best, and corresponding q value
        """
        # make a back-up for current input, current output and returned index of action
        self.last_input = array([state])
        self.target = self.model.predict_on_batch(self.last_input)
        self.last_action = argmax(self.target)
        return self.actions[self.last_action], self.target[self.last_action]

    def update(self, q_value_update):
        # target will be the same as output, except for max value,
        # which will be replaced by q_value_update
        self.target[self.last_action] = q_value_update
        self.model.train_on_batch(self.last_input, array([self.target]))
