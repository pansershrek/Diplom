import numpy as np
import tensorflow as tf
import math

from .base_class import BaseFunction


class TargetFunction1(BaseFunction):

    def __call__(self):
        return (
            self.x[0] ** 3 + self.x[1] ** 2 +
            tf.sin(self.x[0] * math.pi / 180) +
            tf.cos(self.x[1] * math.pi / 180)
        )
