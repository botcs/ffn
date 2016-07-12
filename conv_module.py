from scipy.signal import convolve2d
import numpy as np

import layer_module as lm


class conv(lm._layer):

    def __init__(self, *args, **kwargs):
        self.type = 'conv'
        lm._layer.__init__(self, *args, **kwargs)
        self.kernel_shape = kwargs['kernel_shape']
        self.nok = kwargs.get('num_of_kernels')

        self.kernels = [np.random.randn(*self.kernel_shape)
                        for i in xrange(self.nok)]

    def get_local_output(self, input):
        self.input = np.array(input)
        return [[convolve2d(i, k, mode='valid')
                 for k in self.kernels] for i in input]

    def backprop_delta(self, target):
        self.deltas = self.next_layer.backprop_delta(target)
        return [np.sum([convolve2d(d, k[::-1, ::-1], mode='valid')
                        for k in self.kernels]) for d in self.deltas]

    def train(self, rate):
        for k, d in zip(self.kernels, self.deltas):
            k += -rate * np.sum([convolve2d(i, d, mode='valid')
                                 for i in self.input])


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
