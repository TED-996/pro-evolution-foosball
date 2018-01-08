from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import RMSprop
from numpy import array, expand_dims


class NN:

    def __init__(self, load_file=None,
                 input_dim: int =0,
                 hidden_layers=None,
                 output_dim: int =0):
        """
        :param load_file: a tuple of size 2 with 2 files: one for model and one for NN class remaining attributes
        :param hidden_layers: number of units on each hidden layer
        :param input_dim: input layer dimension(number of units)
        :param output_dim: number of units on output layer
        """
        if load_file is not None:
            self.__load(load_file)
            return
        assert len(hidden_layers) > 0, "NN must contain at least one hidden layer"
        assert output_dim > 0, "NN must contain at least one unit on output layer "

        self.model = Sequential(self.__build_stack_of_layers(input_dim, hidden_layers, output_dim))
        self.compiled = False

    @staticmethod
    def __build_stack_of_layers(input_dim, hidden_layers, output_dim):
        # add first and last layer in network
        nn_layers = [Dense(units=hidden_layers[0],
                           activation="relu",
                           input_dim=input_dim),
                     Dense(units=output_dim)]
        # if there are more hidden layers, add them before last layer
        for layer_size in range(1, len(hidden_layers)):
            nn_layers.insert(-1, Dense(units=layer_size,
                                       activation="relu"))
        return nn_layers

    def __load(self, load_file):
        self.model = load_model(load_file)

    def save(self, file):
        """
        :param file: a tuple of 2 files: one for the model and one for remaining attributes of class NN
        """
        if not self.compiled:
            raise BaseException("Cannot save a model that was not compiled")
        self.model.save(file)

    def compile(self, lr=0.1, rho=0.9, epsilon=1e-08, decay=0.0):
        # TODO improvement of hyper-parameters of NN
        if self.compiled:
            print("already compiled!")
            return
        self.compiled = True
        self.model.compile(optimizer=RMSprop(lr=lr, rho=rho, epsilon=epsilon, decay=decay),
                           loss="mean_squared_error",
                           metrics=["accuracy"])

    def predict_action(self, state):
        """
        :param state: must be a vector with all values that will represent a state
        :return: an array with q values for each possible action
        """
        return self.model.predict_on_batch(array([state]))[0]

    def update(self, state, target):
        """
        :param state: input of the model
        :param target: output of the model
        """
#        for i in range(len(target)):
#            target[i] = expand_dims(target[i], -1)
        self.model.train_on_batch(array(state), array(target))
