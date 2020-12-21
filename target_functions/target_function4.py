import numpy as np
import tensorflow as tf
import math

from .base_class import BaseFunction


class TargetFunction4(BaseFunction):

    def __call__(self):
        return (
            tf.cos((self.x[0] + self.x[1]) * math.pi / 180) /
            (1 + (self.x[0]**4 + self.x[1]**4)**0.5)
        )
