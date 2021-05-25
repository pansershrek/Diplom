import numpy as np
import tensorflow as tf
import math
from sklearn.datasets import load_boston
from utils.utils import convert_variables, convert_variables_without_trainable
from utils.generate_data import generate_set
from utils.utils import get_numpy_array


from .base_class import BaseTargetFunction


class TargetFunction9(BaseTargetFunction):

    val = {}

    def __call__(self):
        """Сalculate function value"""
        if self.x2y:
            return self.x2y[str(get_numpy_array(self.x, 4))]
        return self.val[str(list(get_numpy_array(self.x, 4)))]
