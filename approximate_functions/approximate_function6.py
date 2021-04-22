import numpy as np
import tensorflow as tf
import math

from .base_class import BaseApproximateFunction


class ApproximateFunction6(BaseApproximateFunction):

    def __call__(self, params):
        """Сalculate function value with params

        :param params: params
        :type params: list
        """
        result = 0
        for ind in range(len(self.x)):
            for p in range(3):
                result += params[ind*4+p] * self.x[ind]**p
            result += params[ind*4+3] * tf.cos(self.x[ind] * math.pi / 180)
        return result
