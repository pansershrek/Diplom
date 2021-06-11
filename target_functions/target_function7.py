import numpy as np
import tensorflow as tf
import math

from .base_class import BaseTargetFunction
from utils.utils import get_numpy_array


class TargetFunction7(BaseTargetFunction):

    def __call__(self):
        """Сalculate function value"""
        if self.x2y:
            return self.x2y[str(get_numpy_array(self.x))]
        return (
            self.x[0]**2 + self.x[1]**2 + self.x[2]
            - tf.sin((self.x[2]) * math.pi / 180)
        )


class TargetFunction7_1(BaseTargetFunction):

    def __call__(self):
        """Сalculate function value"""
        if self.x2y:
            return self.x2y[str(get_numpy_array(self.x))]
        return (
            tf.math.exp((self.x[0]**4 + self.x[1]**2) / 100.0) + self.x[2]
        )


class TargetFunction7_2(BaseTargetFunction):

    def __call__(self):
        """Сalculate function value"""
        if self.x2y:
            return self.x2y[str(get_numpy_array(self.x))]
        return (
            - self.x[0] * self.x[1] +
            tf.sin((self.x[2]) * math.pi / 180) + self.x[2]**4
        )
