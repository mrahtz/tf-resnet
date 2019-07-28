import re
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import BatchNormalization, Activation, GlobalAveragePooling2D, Dense

from layers import Conv2DPadReg, ResidualBlock


def make_resnet(net):
    match = re.match(r'([a-z]*)(\d+)', net)
    net_type, n_layers = match.group(1), match.group(2)
    residual = (net_type == 'resnet')
    n = (int(n_layers) - 2) // 6
    print("Net: detected n = {} {} shortcuts".format(n, 'with' if residual else 'without'))

    layers = []
    filters = {
        0: 16,
        1: 32,
        2: 64
    }

    def strides(section_n, block_n, block_idx):
        if section_n >= 1 and block_n == 0 and block_idx == 0:
            return 2
        else:
            return 1

    layers.extend([
        Conv2DPadReg(kernel_size=3, filters=16, input_shape=(32, 32, 3)),
        BatchNormalization(),
        Activation('relu')
    ])
    for section_n in range(3):
        for block_n in range(n):
            block = [
                [
                    Conv2DPadReg(kernel_size=3,
                                 filters=filters[section_n],
                                 strides=strides(section_n, block_n, i)),
                    BatchNormalization(),
                    Activation('relu'),
                ] for i in range(2)
            ]
            block = [x for l in block for x in l]
            if residual:
                layers.append(ResidualBlock(block))
            else:
                layers.extend(block)
    layers.extend([
        GlobalAveragePooling2D(),
        Dense(10, 'softmax')
    ])
    model = Sequential(layers)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    return model
