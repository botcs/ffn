import conv_module as cm
import network_module as nm
import numpy as np

set_size = 5


input_set = np.random.randn(set_size, 1, 10, 10)
target_set = np.eye(set_size)
cnn = nm.network(in_shape=input_set[0, 0].shape, criterion='softmax')
c = cm.Conv(kernel_shape=(5, 5), num_of_ker=10, prev_layer=cnn[0])
cnn.register_new_layer(c)
cnn.add_activation('tanh')
cnn.add_full(set_size)
print cnn


def out(i):
    print cnn.get_output(input_set[i])


def train(e):
    cnn.train(input_set, target_set, e, 0.1)
