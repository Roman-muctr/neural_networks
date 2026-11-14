import numpy as np


class ActivationFunctions:
    def __init__(self):
        raise TypeError("Нельзя создать экземпляр этого класса")

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)
