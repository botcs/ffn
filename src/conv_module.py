from scipy.signal import convolve2d
import numpy as np
import layer_module as lm
import utilities as util

np.set_printoptions(precision=2, edgeitems=2, threshold=5)


class conv(lm._layer):

    def __init__(self, num_of_ker, kernel_shape, *args, **kwargs):
        lm._layer.__init__(self, *args, **kwargs)
        self.type = 'convolution'
        self.kernel_shape = kernel_shape
        self.nok = num_of_ker
        self.kernels = np.random.randn(self.nok, *self.kernel_shape)

        if self.prev_layer:
            '''Shape is the shape of the output layer

               can only be evaluated if the layer is in a network,
               since it depends on it
            '''
            self.shape = (self.nok,)
            self.shape += tuple(np.add(
                np.subtract(self.prev_layer.shape, self.kernel_shape), (1, 1)))
            'For fully connected next layer'
            self.width = np.prod(self.shape)

    def get_local_output(self, input):
        self.input = np.array(input)
        return np.array([np.sum([convolve2d(i, k, mode='valid')for i in input],
                                axis=0) for k in self.kernels])

    def backprop_delta(self, target):
        self.deltas = self.next_layer.backprop_delta(
            target).reshape(self.shape)

        return np.array(
            np.sum([convolve2d(d, k[::-1, ::-1])
                    for d, k in zip(self.deltas, self.kernels)], axis=0))

    def train(self, rate):
        for k, d in zip(self.kernels, self.deltas):
            k += -rate * np.sum([convolve2d(i, d, mode='valid')
                                 for i in self.input])

    def __str__(self):
        res = lm._layer.__str__(self)
        res += '   ->   kernels: {} x {}'.format(
            self.nok, self.kernel_shape)
        return res


class max_pool(lm._layer):

    def __init__(self, shape=None, pool_shape=None, *args, **kwargs):
        lm._layer.__init__(self, shape=shape, type='max pool', *args, **kwargs)
        assert (shape is None) ^ (pool_shape is None),\
            "'pool_shape=' XOR 'shape=' must be defined"

        if self.prev_layer:
            if shape:
                self.pool_shape = np.divide(self.prev_layer.shape, shape)
            else:
                self.shape = np.divide(
                    self.prev_layer.shape, pool_shape).squeeze()
                self.width = np.prod(self.shape)

    def get_local_output(self, input):
        x = np.array(input)
        N, h, w = x.shape
        n, m = self.shape
        x = x.reshape(N, h / n, n, w / m, m)\
             .swapaxes(2, 3)\
             .reshape(N, h / n, w / m, n * m)
        res = np.amax(x, axis=3)
        self.switch = np.any([input == i for i in res.flatten()], axis=0)
        return res

    def backprop_delta(self, target):
        # self.delta = self.next_layer.backprop_delta(target)
        res = np.ones(np.prod(self.pool_shape) + self.shape)
        return res
