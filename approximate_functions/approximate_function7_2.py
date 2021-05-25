import numpy as np
import tensorflow as tf
import math

from .base_class import BaseApproximateFunction


class ApproximateFunction7_2(BaseApproximateFunction):

    def __call__(self, params):
        """Сalculate function value with params

        :param params: params
        :type params: list
        """
        result = 0
        for ind in range(len(self.x)):
            for p in range(0, 10):
                result += params[ind * 10 * 2 + 2 * p] * \
                    tf.sin((self.x[ind] * (p + 1)) * math.pi / 180)
                result += params[ind * 10 * 2 + 2 * p + 1] * \
                    tf.cos((self.x[ind] * (p + 1)) * math.pi / 180)
        result += params[-1]
        return result
