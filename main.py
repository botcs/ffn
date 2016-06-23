import numpy as np
import network_module as nm
from mnist import MNIST

mndata = MNIST('.')
img, lbl = mndata.load_training()

train_data = np.array(img[0:50000], float)
train_data /= train_data.max()

train_label = np.array(lbl[0:50000])
hot_label = np.zeros((train_label.size, train_label.max() + 1))
hot_label[np.arange(train_label.size), train_label] = 1


nn = nm.network(in_size=train_data[0].size, criterion='softmax')
nn.add_full(10)
print 'Training...'
nn.train(input_set=train_data[0:300],
         target_set=hot_label[0:300],
         epoch=1000, rate=0.01)

print nn
