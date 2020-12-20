import numpy as np
import tensorflow as tf

from .base_class import BaseFunction

from utils.utils import convert_variables


class SmothFunction3(BaseFunction):

    @staticmethod
    def get_minimum():
        return [
            convert_variables([-1, -1]),
            convert_variables([1, 1])
        ]

    @staticmethod
    @tf.function
    def __call__(x):
        return (
            (x[0]**6 + x[1] ** 6) - 2 * (x[0]**3 * x[1] + x[0] * x[1]**3) +
            x[0]**4 + x[1]**4 - x[0]**2 + x[1]**2
        )
