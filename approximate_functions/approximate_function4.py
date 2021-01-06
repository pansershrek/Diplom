import numpy as np
import tensorflow as tf
import math

from .base_class import BaseApproximateFunction


class ApproximateFunction4(BaseApproximateFunction):

    def __call__(self, params):
        return (
            params[0] * self.x[0]**2 + params[1] * self.x[0] + params[2] +
            params[3] * self.x[0]**2 + params[4] * self.x[0] + params[5] +
            params[6] * self.x[0] * self.x[1] +
            params[7] * tf.sin(self.x[0] * math.pi / 180) +
            params[8] * tf.sin(self.x[1] * math.pi / 180) +
            params[9] * tf.sin(self.x[0] * math.pi / 180) * self.x[0] +
            params[10] * tf.sin(self.x[0] * math.pi / 180) * self.x[1] +
            params[11] * tf.sin(self.x[1] * math.pi / 180) * self.x[0] +
            params[12] * tf.sin(self.x[1] * math.pi / 180) * self.x[1]
        )
