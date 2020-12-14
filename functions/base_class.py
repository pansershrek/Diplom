import tensorflow as tf


class BaseFunction():

    @staticmethod
    def get_minimum():
        raise None

    @staticmethod
    @tf.function
    def __call__(x):
        raise NotImplementedError
