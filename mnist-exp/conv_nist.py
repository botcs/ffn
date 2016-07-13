import numpy as np
import conv_module as cm
from mnist import MNIST

'must be in the working directory, or in the python path'
import network_module as nm

mndata = MNIST('./MNIST_data')
img, lbl = mndata.load_training()

train_data = np.array(img[:1000], float).reshape(1000, 1, 28, 28)
train_data -= train_data.mean()
train_data /= train_data.std()

train_label = np.array(lbl)
hot_label = np.zeros((train_label.size, train_label.max() + 1))
hot_label[np.arange(train_label.size), train_label] = 1


#########################
# NETWORK DEFINITION
nn = nm.network(in_shape=train_data[0, 0].shape, criterion='softmax')
c = cm.conv(prev_layer=nn[0], num_of_ker=5, kernel_shape=(4, 4))
nn.register_new_layer(c)

nn.add_activation('tanh')
nn.add_full(30)
nn.add_activation('tanh')
nn.add_full(10)
#########################


print 'Training network on MNIST...'
nn.train(input_set=train_data,
         target_set=hot_label,
         epoch=30, rate=0.005)


print 'Validating...'
img, lbl = mndata.load_testing()
test_data = np.array(img, float).reshape(10000, 1, 28, 28)
test_data -= test_data.mean()
test_data /= test_data.std()

test_label = np.array(lbl)
hot_label = np.zeros((test_label.size, test_label.max() + 1))
hot_label[np.arange(test_label.size), test_label] = 1
hit = nn.test_eval(zip(test_data, hot_label))
print 'Hit rate:{}%'.format(hit)

print nn
