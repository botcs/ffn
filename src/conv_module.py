from scipy.signal import convolve2d
import numpy as np
import layer_module as lm

np.set_printoptions(precision=2, edgeitems=5, threshold=5)


class conv(lm._layer):

    def __init__(self, num_of_ker, kernel_shape, **kwargs):
        lm._layer.__init__(self, **kwargs)
        self.type = 'convolution'
        self.kernel_shape = kernel_shape
        self.nok = num_of_ker
        self.kernels = np.random.randn(self.nok, *self.kernel_shape)

        if self.prev:
            self.shape = (self.nok,)
            self.shape += tuple(np.add(np.subtract(
                self.prev.shape[1:], self.kernel_shape), 2*(1,)))
            '''First parameter is the number of corresponding feature maps
            The remaining is the shape of the feature maps

            The output of 2D convolution on an input with
            shape [NxM]
            kernel [kxl]
            results in [(N-k+1) x (M-l+1)]'''

            'For fully connected next layer'
            self.width = np.prod(self.shape)

    def get_local_output(self, input):
        self.input = np.array(input)
        return np.array([np.sum([convolve2d(i, k, mode='valid')for i in input],
                                axis=0) for k in self.kernels])

    def backprop_delta(self, target):
        self.deltas = self.next.backprop_delta(
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

    def __init__(self, pool_shape=None, shape=None, **kwargs):
        lm._layer.__init__(self, shape=shape, type='max pool', **kwargs)
        assert (shape is None) ^ (pool_shape is None),\
            "'pool_shape=' XOR 'shape=' must be defined"

        if self.prev:
            if shape:
                sp = np.divide(self.prev.shape, shape)
                '''First dimension is the number of feature maps in the previous
                   layer'''
                self.pool_shape = tuple(sp[1:])
            else:
                self.pool_shape = pool_shape
                self.shape = tuple(np.divide(self.prev.shape, (1,)+pool_shape))
                self.width = np.prod(self.shape)

    def get_local_output(self, input):
        x = np.array(input)
        N, h, w = x.shape
        n, m = self.pool_shape
        'Reshape for pooling'
        x = x.reshape(N, h / n, n, w / m, m)\
             .swapaxes(2, 3)\
             .reshape(N, h / n, w / m, n * m)
        res = np.amax(x, axis=3)
        'Keep record of which neurons were chosen in the pool by their index'
        self.switch = np.any([input == i for i in res.flatten()], axis=0)\
                        .nonzero()

        return res

    def backprop_delta(self, target):
        self.delta = self.next.backprop_delta(target)
        res = np.zeros(self.prev.shape)
        res[self.switch] = delta.flatten()
        return res
