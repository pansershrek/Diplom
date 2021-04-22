import numpy as np
import tensorflow as tf
import math

from .base_class import BaseApproximateFunction


class ApproximateFunction5(BaseApproximateFunction):

    def __call__(self, params):
        """Сalculate function value with params

        :param params: params
        :type params: list
        """
        result = 0
        for ind in range(len(self.x)):
            for p in range(3):
                result += params[ind*4+p] * self.x[ind]**p
            result += params[ind*4+3] * tf.sin(self.x[ind] * math.pi / 180)
        result +=  params[52] * self.x[0] * tf.sin(self.x[1] * math.pi / 180)
        result +=  params[53] * self.x[0] * tf.sin(self.x[2] * math.pi / 180)
        result +=  params[54] * self.x[0] * tf.sin(self.x[3] * math.pi / 180)
        result +=  params[55] * self.x[0] * tf.sin(self.x[4] * math.pi / 180)
        result +=  params[56] * self.x[0] * tf.sin(self.x[5] * math.pi / 180)

        result +=  params[57] * self.x[1] * tf.sin(self.x[0] * math.pi / 180)
        result +=  params[58] * self.x[1] * tf.sin(self.x[2] * math.pi / 180)
        result +=  params[59] * self.x[1] * tf.sin(self.x[3] * math.pi / 180)
        result +=  params[60] * self.x[1] * tf.sin(self.x[4] * math.pi / 180)
        result +=  params[61] * self.x[1] * tf.sin(self.x[5] * math.pi / 180)

        result +=  params[62] * self.x[2] * tf.sin(self.x[0] * math.pi / 180)
        result +=  params[63] * self.x[2] * tf.sin(self.x[1] * math.pi / 180)
        result +=  params[64] * self.x[2] * tf.sin(self.x[3] * math.pi / 180)
        result +=  params[65] * self.x[2] * tf.sin(self.x[4] * math.pi / 180)
        result +=  params[66] * self.x[2] * tf.sin(self.x[5] * math.pi / 180)

        result +=  params[67] * self.x[3] * tf.sin(self.x[0] * math.pi / 180)
        result +=  params[68] * self.x[3] * tf.sin(self.x[1] * math.pi / 180)
        result +=  params[69] * self.x[3] * tf.sin(self.x[2] * math.pi / 180)
        result +=  params[70] * self.x[3] * tf.sin(self.x[4] * math.pi / 180)
        result +=  params[71] * self.x[3] * tf.sin(self.x[5] * math.pi / 180)

        result +=  params[72] * self.x[4] * tf.sin(self.x[0] * math.pi / 180)
        result +=  params[73] * self.x[4] * tf.sin(self.x[1] * math.pi / 180)
        result +=  params[74] * self.x[4] * tf.sin(self.x[2] * math.pi / 180)
        result +=  params[75] * self.x[4] * tf.sin(self.x[3] * math.pi / 180)
        result +=  params[76] * self.x[4] * tf.sin(self.x[5] * math.pi / 180)

        result +=  params[77] * self.x[5] * tf.sin(self.x[0] * math.pi / 180)
        result +=  params[78] * self.x[5] * tf.sin(self.x[1] * math.pi / 180)
        result +=  params[79] * self.x[5] * tf.sin(self.x[2] * math.pi / 180)
        result +=  params[80] * self.x[5] * tf.sin(self.x[3] * math.pi / 180)
        result +=  params[81] * self.x[5] * tf.sin(self.x[5] * math.pi / 180)
        return result
