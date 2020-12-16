import numpy as np
import tensorflow as tf

from .base_class import BaseFunction

from utils.utils import convert_variables


class SmothFunction2(BaseFunction):

    @staticmethod
    def get_minimum():
        return [convert_variables([1, 1])]

    @staticmethod
    @tf.function
    def __call__(x):
        return (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
