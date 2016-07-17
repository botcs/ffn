import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt

'must be in the working directory, or in the python path'
import network_module as nm

print 'Loading data from MNIST'
mndata = MNIST('./MNIST_data')
img, lbl = mndata.load_training()

train_data = np.array(img, float).reshape(60000, 1, 28, 28)
train_data -= train_data.mean()
train_data /= train_data.std()

train_label = np.array(lbl)
train_hot = np.zeros((train_label.size, train_label.max() + 1))
train_hot[np.arange(train_label.size), train_label] = 1

img, lbl = mndata.load_testing()
test_data = np.array(img, float).reshape(10000, 1, 28, 28)
test_data -= test_data.mean()
test_data /= test_data.std()

test_label = np.array(lbl)
test_hot = np.zeros((test_label.size, test_label.max() + 1))
test_hot[np.arange(test_label.size), test_label] = 1

print 'constructing network'
#########################
# NETWORK DEFINITION
nn = nm.network(in_shape=train_data[0].shape, criterion='softmax')
# nn.add_conv(10, (5, 5))
# nn.add_activation('tanh')
# nn.add_maxpool((4, 4))
nn.add_full(650)
nn.add_activation('tanh')
nn.add_full(10)
#########################
print nn


def print_csv(filename, data):
    with open(filename, 'wb') as out:
        for t in data:
            out.write('{}\t{}\n'.format(*t))

name = 'FC_650-10'


print 'Training network on MNIST...'
result = nn.train(input_set=train_data,
                  target_set=train_hot,
                  epoch=30, rate=0.005,
                  test_set=(zip(test_data, test_hot)),
                  checkpoint='./test_runs/nets/{}-rate005'.format(name))

print_csv('./test_runs/{}-rate005'.format(name), result)
