import numpy as np
import tensorflow as tf
import math
from sklearn.datasets import load_boston
from utils.utils import convert_variables, convert_variables_without_trainable
from utils.generate_data import generate_set
from utils.utils import get_numpy_array

from sklearn.datasets import load_diabetes

from .base_class import BaseTargetFunction


class TargetFunction1(BaseTargetFunction):

    def __init__(self):
        self.val = {}
        X, y = load_boston(return_X_y=True)
        for k, v in zip(X, y):
            self.val[str(convert_variables_without_trainable(k))] = v

    def __call__(self):
        """Сalculate function value"""
        return self.val[str(self.x)]
