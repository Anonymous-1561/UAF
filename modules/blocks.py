from .convs import conv1d
from .others import layer_norm, gelu
import tensorflow as tf


def res_block(input_, dilation, layer_id, residual_channels, kernel_size, causal=True, train=True):
    block_name = "residual_{}_{}".format(layer_id, dilation)
    with tf.variable_scope(block_name, reuse=tf.AUTO_REUSE):
        x = conv1d(input_, residual_channels, dilation, kernel_size, causal=causal, name="dilated_conv1")
        x = layer_norm(x, name="layer_norm1", trainable=train)
        x = tf.nn.relu(x)

        x = conv1d(x, residual_channels, 2 * dilation, kernel_size, causal=causal, name="dilated_conv2")
        x = layer_norm(x, name="layer_norm2", trainable=train)
        x = tf.nn.relu(x)

        return input_ + x


def res_block_freeze(
    input_, dilation, layer_id, residual_channels, kernel_size, causal=True, train=True, finetune=False
):
    if not finetune:
        block_name = "residual_{}_{}".format(layer_id, dilation)  # Original Block
    else:
        block_name = "finetune_residual_{}_{}".format(layer_id, dilation)  # Fine-Tune block

    with tf.variable_scope(block_name, reuse=tf.AUTO_REUSE):
        x = conv1d(input_, residual_channels, dilation, kernel_size, causal=causal, name="dilated_conv1")
        x = layer_norm(x, name="layer_norm1", trainable=train)
        x = tf.nn.relu(x)

        x = conv1d(x, residual_channels, 2 * dilation, kernel_size, causal=causal, name="dilated_conv2")
        x = layer_norm(x, name="layer_norm2", trainable=train)
        x = tf.nn.relu(x)

        return input_ + x


def peter_block(
    input_, dilation, layer_id, residual_channels, kernel_size, causal=True, train=True, adapter=True, cardinality=32
):
    block_name = "residual_{}_{}".format(layer_id, dilation)
    with tf.variable_scope(block_name, reuse=tf.AUTO_REUSE):
        x = conv1d(input_, residual_channels, dilation, kernel_size, causal=causal, name="dilated_conv1")
        if adapter:
            x = get_adapter_split_trans_aggr_with_name(x, cardinality, name="adapters_1")
        x = layer_norm(x, name="layer_norm1", trainable=train)
        x = tf.nn.relu(x)

        x = conv1d(x, residual_channels, 2 * dilation, kernel_size, causal=causal, name="dilated_conv2")
        if adapter:
            x = get_adapter_split_trans_aggr_with_name(x, cardinality, name="adapters_2")
        x = layer_norm(x, name="layer_norm2", trainable=train)
        x = tf.nn.relu(x)

        return input_ + x


def get_adapter_split_trans_aggr_with_name(input_, cardinality=32, name="adapters"):
    with tf.variable_scope(name):
        residual_channels = input_.get_shape()[-1]
        hidden_size = residual_channels / (cardinality * 4)

        block_sets = list()
        for i in range(cardinality):
            conv_down_i = conv1d(input_, hidden_size, name="adapter_conv1_down_{}".format(i))
            conv_down_i = gelu(conv_down_i)
            conv_up_i = conv1d(conv_down_i, residual_channels, name="adapter_conv1_up_{}".format(i))
            block_sets.append(conv_up_i)

        output = tf.add_n(block_sets)
        return input_ + output
