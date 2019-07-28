from functools import partial

import tensorflow as tf
from tensorflow.python.keras.layers import Activation, Conv2D
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.layers.base import Layer

Conv2DPadReg = partial(Conv2D, padding='same', kernel_regularizer=l2(1e-4))


class ResidualBlock(Layer):
    def __init__(self, layers):
        super().__init__()
        assert isinstance(layers[-1], Activation)
        self.layers = layers

    def call(self, x, **kwargs):
        y = x
        for layer in self.layers[:-1]:
            y = layer(y)
        if x.shape.as_list() != y.shape.as_list():
            x = self._reshape(x, y.shape.as_list())
        out = tf.add(x, y)
        out = self.layers[-1](out)  # Activation
        return out

    def _reshape(self, x, y_shape):
        # Downsample feature map by dropping pixels
        # Upsample channels using a learned transformation
        size_in, size_out = x.shape.as_list()[1], y_shape[1]
        downsample = size_in // size_out
        n_channels = y_shape[3]
        x = Conv2D(kernel_size=1, filters=n_channels, strides=downsample)(x)
        return x
