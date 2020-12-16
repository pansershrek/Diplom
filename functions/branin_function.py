import numpy as np
import tensorflow as tf
import math

from .base_class import BaseFunction

from utils.utils import convert_variables


class BraninFunction(BaseFunction):

    @staticmethod
    def get_minimum():
        return [
            convert_variables([-math.pi, 12.275]),
            convert_variables([math.pi, 2.275]),
            convert_variables([9.42478, 2.475]),
        ]

    @staticmethod
    def __call__(x):
        return (
            (x[1] - 5.1 * x[0]**2 / (4 * math.pi ** 2) + 5 * x[0] / math.pi - 6)**2 +
            10 * (1 - 1 / (8 * math.pi)) * math.cos(math.radians(x[0].numpy()))
            + 10
        )
