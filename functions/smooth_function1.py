import numpy as np
import tensorflow as tf

from .base_class import BaseFunction

from utils.utils import convert_variables


class SmothFunction1(BaseFunction):

    @staticmethod
    def get_minimum():
        return convert_variables([-1, -1])

    @staticmethod
    @tf.function
    def __call__(x):
        return 1 + x[0] + x[1] - x[0] * x[1] + x[0] ** 2 + x[1] ** 2
