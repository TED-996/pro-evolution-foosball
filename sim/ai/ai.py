from numpy import array, arange
from itertools import product
from .NN import NN
import pickle


class AI:
    def __init__(self, load: bool = False,
                 state_size: int = 0,
                 rods_number: int = 0,
                 offset=None,
                 angle_velocity=None,
                 hidden_layers=(100, 100),
                 nn_file: str = "save.model",
                 actions_file: str = "save.actions"):
        self.actions = None
        self.model = None
        self.lamda = 0.9  # TODO adjust
        self.alpha = 0.2  # TODO adjust
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

