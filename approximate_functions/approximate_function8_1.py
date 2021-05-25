import numpy as np
import tensorflow as tf
import math

from .base_class import BaseApproximateFunction


class ApproximateFunction8_1(BaseApproximateFunction):

    def __call__(self, params):
        """Сalculate function value with params

        :param params: params
        :type params: list
        """
        result = 0
        for ind in range(len(self.x)):
            for p in range(1, 5):
                result += params[ind * 4 + p - 1] * self.x[ind]**p
        result += params[-1]
        return result
