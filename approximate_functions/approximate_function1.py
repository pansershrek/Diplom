import numpy as np
import tensorflow as tf
import math

from .base_class import BaseApproximateFunction


class ApproximateFunction1(BaseApproximateFunction):

    def __call__(self, params):
        """Сalculate function value with params

        :param params: params
        :type params: list
        """
        return (
            params[0] * self.x[0]**3 + params[1] * self.x[0]**2 + params[2] * self.x[0] + params[3] +
            params[4] * self.x[1]**3 + params[5] * self.x[0]**2 + params[6] * self.x[0] + params[7] +
            params[8] * self.x[0]**2 * self.x[1] + params[9] * self.x[0] * self.x[1] ** 2 +
            params[10] * self.x[0] * self.x[1]
        )
