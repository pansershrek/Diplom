import numpy as np
import tensorflow as tf
import math

from .base_class import BaseFunction


class ApproximateFunction1(BaseFunction):

    def __call__(self, params):
        return self._linear_transformation(params)
