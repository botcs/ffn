import numpy as np
import network_module as nm
from mnist import MNIST

mndata = MNIST('.')
img, lbl = mndata.load_training()

train_data = np.array(img, float)
train_data -= train_data.mean()
train_data /= train_data.std()

train_label = np.array(lbl)
hot_label = np.zeros((train_label.size, train_label.max() + 1))
hot_label[np.arange(train_label.size), train_label] = 1


nn = nm.network(in_size=train_data[0].size, criterion='softmax')
nn.add_full(10)
print 'Training network on MNIST...'
nn.train(input_set=train_data,
         target_set=hot_label,
         epoch=100, rate=0.01,
         checkpoint='MNIST')


print 'Validating...'
img, lbl = mndata.load_testing()
test_data = np.array(img, float)
test_data -= train_data.mean()
test_data /= train_data.std()

test_label = np.array(lbl)
hot_label = np.zeros((train_label.size, train_label.max() + 1))
hot_label[np.arange(train_label.size), train_label] = 1
hit = 100 * nn.classification_validate(test_data, hot_label)
print 'Hit rate:{}%'.format(hit)

print nn
