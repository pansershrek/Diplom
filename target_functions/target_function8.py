import numpy as np
import tensorflow as tf
import math

from .base_class import BaseTargetFunction
from utils.utils import get_numpy_array


class TargetFunction8(BaseTargetFunction):

    def __call__(self):
        """Сalculate function value"""
        if self.x2y:
            return self.x2y[str(get_numpy_array(self.x))]
        return (self.x[0]**3 + tf.sin((self.x[0] + self.x[1]) * math.pi / 180) + math.e**(-self.x[2]) + 1 / (self.x[3]**2 + 4))
