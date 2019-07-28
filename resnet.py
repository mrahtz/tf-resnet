import re
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense

from layers import ConvBNRelu, ConvBNReluResidualBlock, ConvBNReluBlock


def make_resnet(net):
    n, residual = parse_net(net)
    print("Net: detected n = {} {} shortcuts".format(n, 'with' if residual else 'without'))

    Block = ConvBNReluResidualBlock if residual else ConvBNReluBlock
    layers = [
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
    layers = flatten_layers(layers)

    model = Sequential(layers)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    return model


def flatten_layers(_layers):
    layers = []
    for x in _layers:
        try:
            for l in x:
                layers.append(l)
        except TypeError:
            layers.append(x)
    return layers


def parse_net(net):
    match = re.match(r'([a-z]*)(\d+)', net)
    net_type, n_layers = match.group(1), match.group(2)
    residual = (net_type == 'resnet')
    n = (int(n_layers) - 2) // 6
    return n, residual
