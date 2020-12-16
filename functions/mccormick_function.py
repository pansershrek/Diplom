import numpy as np
import tensorflow as tf
import math

from .base_class import BaseFunction

from utils.utils import convert_variables


class McCormickFunction(BaseFunction):

    @staticmethod
    def get_minimum():
        return [
            convert_variables([-0.547198, -1.5472]),
        ]

    @staticmethod
    def __call__(x):
        return (
            math.sin(math.radians(x[0].numpy() + x[1].numpy())) +
            (x[0] - x[1])**2 - 1.5 * x[0] + 2.5 * x[1] + 1
        )
