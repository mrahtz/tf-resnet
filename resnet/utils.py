import re

import os
from tensorflow.python.util import deprecation


def tf_disable_warnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def tf_disable_deprecation_warnings():
    deprecation._PRINT_DEPRECATION_WARNINGS = False


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