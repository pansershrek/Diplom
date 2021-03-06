import numpy as np
import tensorflow as tf
import math

from .base_class import BaseApproximateFunction


class ApproximateFunction8_3(BaseApproximateFunction):

    def __call__(self, params):
        """Сalculate function value with params

        :param params: params
        :type params: list
        """
        result = 0
        for ind in range(len(self.x)):
            for p in range(0, 3):
                result += params[ind * 6 + 2 * p] * \
                    tf.math.exp(params[ind * 6 + 2 * p + 1]) * self.x[ind]
        result += params[-1]
        return result
