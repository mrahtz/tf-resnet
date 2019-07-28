import re
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, Conv2D

from layers import ConvBNRelu, ConvBNReluResidualBlock, ConvBNReluBlock


def make_resnet(net):
    match = re.match(r'([a-z]*)(\d+)', net)
    net_type, n_layers = match.group(1), match.group(2)
    residual = (net_type == 'resnet')
    n = (int(n_layers) - 2) // 6
    print("Net: detected n = {} {} shortcuts".format(n, 'with' if residual else 'without'))

    Block = ConvBNReluResidualBlock if residual else ConvBNReluBlock
    _layers = [
        ConvBNRelu(strides=1, filters=16, input_shape=(32, 32, 3)),

        Block(strides=1, filters=16),
        [Block(strides=1, filters=16) for _ in range(n - 1)],

        Block(strides=2, filters=32),
        [Block(strides=1, filters=32) for _ in range(n - 1)],

        Block(strides=2, filters=64),
        [Block(strides=1, filters=64) for _ in range(n - 1)],

        GlobalAveragePooling2D(),
        Dense(10, 'softmax')
    ]

    layers = []
    for x in _layers:
        try:
            for l in x:
                layers.append(l)
        except TypeError:
            layers.append(x)

    model = Sequential(layers)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    return model
