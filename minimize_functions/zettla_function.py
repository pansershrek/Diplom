import numpy as np
import tensorflow as tf

from .base_class import BaseFunction

from utils.utils import convert_variables


class ZettlaFunction(BaseFunction):

    @staticmethod
    def get_minimum():
        return [convert_variables([-0.029896, 0])]

    @staticmethod
    @tf.function
    def __call__(x):
        return 0.25 * x[0] + (x[0] * x[0] - 2 * x[0] + x[1] ** 2) ** 2
