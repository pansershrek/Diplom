import numpy as np
import tensorflow as tf

from .base_class import BaseFunction

from utils.utils import convert_variables


class MatyasFunction(BaseFunction):

    @staticmethod
    def get_minimum():
        return convert_variables([0, 0])

    @staticmethod
    @tf.function
    def __call__(x):
        return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]
