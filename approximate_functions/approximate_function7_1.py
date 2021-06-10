import numpy as np
import tensorflow as tf
import math

from .base_class import BaseApproximateFunction


class ApproximateFunction7_1(BaseApproximateFunction):

    P = 5

    def __call__(self, params):
        """Сalculate function value with params

        :param params: params
        :type params: list
        """
        result = 0
        for ind in range(len(self.x)):
            for p in range(0, self.P):
                result += params[ind * self.P + p] * self.x[ind]**p
        result += params[-1]
        return result
