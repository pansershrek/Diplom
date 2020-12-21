import numpy as np
import tensorflow as tf
import math

from .base_class import BaseFunction


class TargetFunction2(BaseFunction):

    def __call__(self):
        return (
            tf.sin((self.x[0] + self.x[1]) * math.pi / 180) +
            (self.x[0] - self.x[1])**2 - 1.5 * self.x[0] + 2.5 * self.x[1] + 1
        )
