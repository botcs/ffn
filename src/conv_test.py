import network_module as nm
import numpy as np

set_size = 100


input_set = np.random.randn(set_size, 1, 44, 34)
target_set = np.eye(set_size)
cnn = nm.network(in_shape=input_set[0].shape, criterion='softmax')
cnn.add_conv(3, (5, 5))
cnn.add_activation('tanh')
cnn.add_conv(2, (5, 5))
cnn.add_activation('tanh')
cnn.add_conv(1, (5, 5))
cnn.add_activation('tanh')
cnn.add_maxpool((2, 2))
cnn.add_full(set_size)
print cnn


def out(i):
    return cnn.get_output(input_set[i])


def train(e):
    cnn.train(input_set, target_set, e, 0.1)
