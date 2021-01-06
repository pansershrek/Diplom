import numpy as np
import tensorflow as tf
import math

from .base_class import BaseTargetFunction


class TargetFunction3(BaseTargetFunction):

    def __call__(self):
        """Сalculate function value"""
        return (
            tf.math.log(1 + self.x[0]**4 + self.x[1]**4) +
            tf.sin((self.x[0] + self.x[1]) * math.pi / 180) + self.x[0]**3
        )
