import numpy as np
import tensorflow as tf
import math

from .base_class import BaseApproximateFunction


class ApproximateFunction3(BaseApproximateFunction):

    def __call__(self, params):
        """Сalculate function value with params

        :param params: params
        :type params: list
        """
        return (
            params[0] * self.x[0]**3 + params[1] * self.x[0]**2 + params[2] * self.x[0] + params[3] +
            params[4] * self.x[1]**3 + params[5] * self.x[0]**2 + params[6] * self.x[0] + params[7] +
            params[8] * self.x[0]**2 * self.x[1] + params[9] * self.x[0] * self.x[1] ** 2 +
            params[10] * self.x[0] * self.x[1] + params[11] * tf.sin(self.x[0] * math.pi / 180) +
            params[12] * tf.sin(self.x[1] * math.pi / 180) +
            params[13] * tf.sin((self.x[0] + self.x[1]) * math.pi / 180) +
            params[14] * tf.sin(self.x[0] * math.pi / 180) * self.x[0] +
            params[15] * tf.sin(self.x[0] * math.pi / 180) * self.x[1] +
            params[16] * tf.sin(self.x[1] * math.pi / 180) * self.x[0] +
            params[17] * tf.sin(self.x[1] * math.pi / 180) * self.x[1] +
            params[18] * tf.sin(self.x[0] * math.pi / 180) * self.x[0]**2 +
            params[19] * tf.sin(self.x[0] * math.pi / 180) * self.x[1]**2 +
            params[20] * tf.sin(self.x[1] * math.pi / 180) * self.x[0]**2 +
            params[21] * tf.sin(self.x[1] * math.pi / 180) * self.x[1]**2
        )
