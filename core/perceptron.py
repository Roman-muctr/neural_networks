import numpy as np


class Perceptron:
    _name: str
    _flag_training: bool

    def __init__(self, layer_sizes: list, seed: int = 42): #2 3 1
        self._rng = np.random.default_rng(seed)
        self._sizes = layer_sizes
        self.biases = [self._rng.uniform(-0.5, 0.5, size=(y, 1)) for y in layer_sizes[1:]]
        self.weights = [self._rng.uniform(-0.5, 0.5, size=(y, x)) for y, x in zip(layer_sizes[1:], layer_sizes[:-1])]
