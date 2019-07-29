from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense

from resnet.layers import ConvBNRelu, ConvBNReluResidualBlock, ConvBNReluBlock
from resnet.utils import flatten_layers, parse_net


def make_resnet(net):
    n, residual = parse_net(net)
    print("Net: detected n = {} {} shortcuts".format(n, 'with' if residual else 'without'))

    Block = ConvBNReluResidualBlock if residual else ConvBNReluBlock
    layers = [
        ConvBNRelu(strides=1, kernel_size=3, filters=16, input_shape=(32, 32, 3)),

        Block(strides=1, kernel_size=3, filters=16),
        [Block(strides=1, kernel_size=3, filters=16) for _ in range(n - 1)],

        Block(strides=2, kernel_size=3, filters=32),
        [Block(strides=1, kernel_size=3, filters=32) for _ in range(n - 1)],

        Block(strides=2, kernel_size=3, filters=64),
        [Block(strides=1, kernel_size=3, filters=64) for _ in range(n - 1)],

        GlobalAveragePooling2D(),
        Dense(10, 'softmax')
    ]
    layers = flatten_layers(layers)

    model = Sequential(layers)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    return model
