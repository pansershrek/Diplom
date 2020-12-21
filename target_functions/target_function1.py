import numpy as np
import tensorflow as tf
import math

from .base_class import BaseFunction


class TargetFunction1(BaseFunction):

    def __call__(self):
        return (
            3 * self.x[0] + 4 + 2 * self.x[1] + 7
        )
