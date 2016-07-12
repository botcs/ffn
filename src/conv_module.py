from scipy.signal import convolve2d
import numpy as np
import layer_module as lm

np.set_printoptions(precision=2, edgeitems=2, threshold=5)


class conv(lm._layer):

    def __init__(self, *args, **kwargs):
        lm._layer.__init__(self, *args, **kwargs)
        self.type = 'conv'
        self.kernel_shape = kwargs['kernel_shape']
        self.nok = kwargs.get('num_of_ker')
        self.kernels = np.random.randn(self.nok, *self.kernel_shape)

        self.shape = np.add(
            np.subtract(self.prev_layer.shape, self.kernel_shape), (1, 1))

        'For fully connected next layer'
        self.width = self.nok * np.prod(self.shape)

    def get_local_output(self, input):
        self.input = np.array(input)
        return np.array([np.sum([convolve2d(i, k, mode='valid')for i in input],
                                axis=0) for k in self.kernels])

    def backprop_delta(self, target):
        self.deltas = self.next_layer.backprop_delta(target)
        return [np.sum([convolve2d(d, k[::-1, ::-1], mode='valid')
                        for k in self.kernels]) for d in self.deltas]

    def train(self, rate):
        for k, d in zip(self.kernels, self.deltas):
            k += -rate * np.sum([convolve2d(i, d, mode='valid')
                                 for i in self.input])

    def __str__(self):
        res = 'CONVOLUTION: {} x [{} x {}] kernel'.format(
            self.nok, *self.kernel_shape)
        return res


class max_pool(lm._layer):

    def __init__(self, *args, **kwargs):
        lm._layer.__init__(self, *args, **kwargs)
        self.type = 'max_pool'
        self.shape = kwargs['shape']

    def get_local_output(self, input):
        x = np.array(input)
        N, h, w = x.shape
        n, m = self.shape
        x = x.reshape(N, h / n, n, w / m, m)\
            .swapaxes(2, 3)\
            .reshape(N, h / n, w / m, n * m)
        return np.amax(x, axis=3)
