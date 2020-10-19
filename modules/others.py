import tensorflow as tf
import numpy as np


def layer_norm(x, name, epsilon=1e-8, trainable=True):
    with tf.variable_scope(name):
        shape = x.get_shape()
        beta = tf.get_variable("beta", [int(shape[-1])], initializer=tf.constant_initializer(0), trainable=trainable)
        gamma = tf.get_variable("gamma", [int(shape[-1])], initializer=tf.constant_initializer(1), trainable=trainable)

        mean, variance = tf.nn.moments(x, axes=[len(shape) - 1], keep_dims=True)
        x = (x - mean) / tf.sqrt(variance + epsilon)

        return gamma * x + beta


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf
