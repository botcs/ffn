import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt

'must be in the working directory, or in the python path'
import network_module as nm


def loadmnist():
    global train_data
    global train_hot
    global test_data
    global test_hot
    print 'Loading data from MNIST'
    mndata = MNIST('./MNIST_data')
    train_size = 60000
    test_size = 1000
    img, lbl = mndata.load_training()

    train_data = np.array(img[0:train_size], float).reshape(-1, 1, 28, 28)
    train_data -= train_data.mean()
    train_data /= train_data.std()

    train_label = np.array(lbl[0:train_size])
    train_hot = np.zeros((train_label.size, train_label.max() + 1))
    train_hot[np.arange(train_label.size), train_label] = 1

    img, lbl = mndata.load_testing()
    test_data = np.array(img[0:test_size], float).reshape(-1, 1, 28, 28)
    test_data -= test_data.mean()
    test_data /= test_data.std()

    test_label = np.array(lbl[0:test_size])
    test_hot = np.zeros((test_label.size, test_label.max() + 1))
    test_hot[np.arange(test_label.size), test_label] = 1

loadmnist()

print 'constructing network'
#########################
# NETWORK DEFINITION
nn = nm.network(in_shape=train_data[0].shape, criterion='MSE')
nn.add_conv(3, (5, 5))
nn.add_maxpool(pool_shape=(2, 2))
nn.add_activation('tanh')

# nn.add_full(150)
nn.add_shaper(np.prod(nn[-1].shape))
nn.add_full(10)
#########################
print nn


def print_csv(filename, data):
    with open(filename, 'wb') as out:
        for t in data:
            out.write('{}\t{}\n'.format(*t))

name = 'conv_10x5x5--FC_10'


print 'Training network on MNIST...'
# old train
# result = nn.train(input_set=train_data,
#                   target_set=train_hot,
#                   epoch=30, rate=0.005,
#                   test_set=(zip(test_data, test_hot)),
#                   checkpoint='./test_runs/nets/{}-rate005'.format(name))

result = []


def print_test():
    print nn.last_epoch, ' ', nn.test_eval((test_data, test_hot))
    result.append((nn.test_eval((train_data, train_hot)),
                   nn.test_eval((test_data, test_hot))))


def imshow(im):
    import matplotlib.pyplot as plt
    if len(im.shape) == 3:
        for i, x in enumerate(im, 1):
            plt.subplot(1, len(im), i)
            plt.imshow(x.squeeze(), cmap='Greys_r')
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
    if len(im.shape) == 4:
        for irow, xrow in enumerate(im, 0):
            for icol, x in enumerate(xrow, 1):
                print '\r  ', len(im), len(xrow), irow * len(xrow) + icol
                plt.subplot(len(im), len(xrow), irow * len(xrow) + icol)
                plt.imshow(x.squeeze(), cmap='Greys_r', interpolation='None')
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.show()

    
def onebatch():
    nn.get_output(train_data[0:10])
    nn.input.backprop_delta(train_hot[0:10])
    nn[1].SGDtrain(0.05)
    nn[4].SGDtrain(0.05)
    nn[7].SGDtrain(0.05)
    return nn.test_eval((train_data[0:10], train_hot[0:10]))
    
    
# imshow(nn.get_output(test_data[0]))
# test = nn.get_output(test_data[0:3]) * 100
# im = zip(test_data[0:3], test, nn[1].backprop_delta(test))
# imshow(im[2])

# nn.SGD(train_policy=nn.fix_epoch, training_set=(train_data, train_hot),
#        batch=16, rate=0.005, epoch_call_back=print_test, epoch=10)

# print_csv('./test_runs/{}-rate005'.format(name), result)

def __main__():
    loadmnist()

if __name__ == '__main__':
    main()

test = nn.grad_ascent(3, train_data[0:100])
