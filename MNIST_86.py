import numpy as np
import network_module as nm
from mnist import MNIST
import pdb

mndata = MNIST('./MNIST')
img, lbl = mndata.load_training()

train_data = np.array(img, float)
train_data -= train_data.mean()
train_data /= train_data.std()

train_label = np.array(lbl)
hot_label = np.zeros((train_label.size, train_label.max() + 1))
hot_label[np.arange(train_label.size), train_label] = 1


#########################
# NETWORK DEFINITION
nn = nm.network(in_size=train_data[0].size, criterion='softmax')
nn.add_full(100)
nn.add_activation(type='logistic')
nn.add_full(10)
#########################


print 'Training network on MNIST...'
nn.train(input_set=train_data,
         target_set=hot_label,
         epoch=10, rate=0.1)


print 'Validating...'
img, lbl = mndata.load_testing()
test_data = np.array(img, float)
test_data -= test_data.mean()
test_data /= test_data.std()

test_label = np.array(lbl)
hot_label = np.zeros((test_label.size, test_label.max() + 1))
hot_label[np.arange(test_label.size), test_label] = 1
hit = 100 * nn.test_eval(zip(test_data, hot_label))
print 'Hit rate:{}%'.format(hit)

print nn
