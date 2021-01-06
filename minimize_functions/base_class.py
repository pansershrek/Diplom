import tensorflow as tf


class BaseFunction():

    @staticmethod
    def get_minimum():
        """Get list of function minimimums

        :return: list of function minimimums
        :rtype: list
        """
        return None

    @staticmethod
    @tf.function
    def __call__(x):
        raise NotImplementedError
