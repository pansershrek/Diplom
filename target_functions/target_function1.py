import numpy as np
import tensorflow as tf
import math

from .base_class import BaseTargetFunction


class TargetFunction1(BaseTargetFunction):

    def __call__(self):
        """Сalculate function value"""
        return (
            1 - 8 * self.x[0] + 7 * self.x[0]**2 -
            7 / 3 * self.x[0]**3 + 1 / 4 * self.x[0]**4
        ) * self.x[1]**2 * math.e**(-self.x[0])
