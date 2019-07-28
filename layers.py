import tensorflow as tf
from tensorflow.python.keras import Sequential, Model
from tensorflow.python.keras.layers import Activation, Conv2D, BatchNormalization
from tensorflow.python.keras.regularizers import l2


class ConvBNRelu(Sequential):
    def __init__(self, filters, strides, skip_relu=False, input_shape=None):
        conv2d_kwargs = {'input_shape': input_shape} if input_shape is not None else {}
        layers = [
            Conv2D(kernel_size=3, filters=filters, strides=strides,
                   padding='same', kernel_regularizer=l2(1e-4), **conv2d_kwargs),
            BatchNormalization(),
        ]
        if not skip_relu:
            layers.append(Activation('relu'))
        super().__init__(layers)


class ResidualBlock(Model):
    def __init__(self, layers, activation):
        super().__init__()
        self._layers = layers
        self.activation = activation
        self._called = False

    def call(self, inputs, **kwargs):
        # We should create the reshape ops in init(), but we're lazy and just do everything in call(),
        # so we should be careful to avoid being called more than once
        assert not self._called

        x = inputs
        for layer in self._layers:
            x = layer(x)
        if inputs.shape.as_list() != x.shape.as_list():
            inputs = self._reshape(inputs, x.shape.as_list())
        x = tf.add(inputs, x)
        x = Activation(self.activation)(x)

        self._called = True
        return x

    def _reshape(self, inputs, x_shape):
        # Downsample feature map by dropping pixels
        # Upsample channels using a learned transformation
        size_in, size_out = inputs.shape.as_list()[1], x_shape[1]
        downsample = size_in // size_out
        n_channels = x_shape[3]
        inputs = Conv2D(kernel_size=1, filters=n_channels, strides=downsample)(inputs)
        return inputs


class ConvBNReluResidualBlock(ResidualBlock):
    def __init__(self, filters, strides):
        layers = [
            ConvBNRelu(filters=filters, strides=strides),
            ConvBNRelu(filters=filters, strides=strides, skip_relu=False),
        ]
        super().__init__(layers, activation='relu')


class ConvBNReluBlock(Sequential):
    def __init__(self, filters, strides):
        layers = [
            ConvBNRelu(filters=filters, strides=strides),
            ConvBNRelu(filters=filters, strides=strides)
        ]
        super().__init__(layers)
