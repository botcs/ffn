from scipy.signal import convolve2d
import numpy as np
import layer_module as lm

np.set_printoptions(precision=2, edgeitems=2, threshold=5)


class Conv(lm.AbstractLayer):

    def __init__(self, num_of_featmap, kernel_shape, **kwargs):
        lm.AbstractLayer.__init__(self, **kwargs)
        self.type = 'convolution'
        self.kernel_shape = kernel_shape
        self.nof = num_of_featmap
        "prev layer's shape[0] is the number of output channels/feature maps"
        self.kernels = np.random.randn(
            self.prev.shape[0], self.nof, *self.kernel_shape)

        if self.prev:
            self.shape = (self.nof,)
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
            self.bias = np.random.randn(*self.shape)

    def get_local_output(self, input):
        assert type(input) == np.ndarray

        'input will be required for training'
        if len(input.shape) == 3:
            'if input is a single sample, extend it to a 1 sized batch'
            self.input = np.expand_dims(input, 0)
        else:
            self.input = input

        '''Each channel has its corresponding kernel in each feature map
        feature map activation is evaluated by summing their activations
        for each sample in input batch'''
        return np.sum(
            [[[convolve2d(channel, kernel, 'valid')
               for kernel in kernel_set]
              for channel, kernel_set in zip(sample, self.kernels)]
             for sample in self.input], axis=1) + self.bias

    def backprop_delta(self, target):
        self.delta = self.next.backprop_delta(target).reshape(self.shape)

        '''Each feature map is the result of all previous layer maps,
        therefore the same gradient has to be spread for each'''
        return np.sum([
            [[convolve2d(k[::-1, ::-1], d) for k in kernel_set]
             for d, kernel_set in zip(sample_delta, self.kernels)]
            for sample_delta in self.delta], axis=1)

    def get_param_grad(self):
        return (np.array(
            # KERNEL GRAD
            [[[convolve2d(channel, d, 'valid')
               for d in sample_delta]
              for channel in sample_input]
             for sample_input, sample_delta in zip(self.input, self.delta)]),
            # BIAS GRAD
            self.delta)

    def SGDtrain(self, rate):
        k_update, b_update = self.get_param_grad()
        self.kernels -= rate * k_update.mean()
        self.bias -= rate / b_update.mean()

    def __str__(self):
        res = lm.AbstractLayer.__str__(self)
        res += '   ->   kernels: {}'.format(self.kernels.shape)
        return res


class max_pool(lm.AbstractLayer):

    def __init__(self, pool_shape=None, shape=None, **kwargs):
        lm.AbstractLayer.__init__(self, shape=shape, type='max pool', **kwargs)
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

        if len(input.shape) == 3:
            'if input is a single sample, extend it to a 1 sized batch'
            x = np.expand_dims(input, 0)
        else:
            x = input

        'batch size will be needed for backprop'
        self.batch, N, h, w = x.shape
        n, m = self.pool_shape
        'Reshape for pooling'
        res = input.reshape(self.batch, N, h/n, n, w/m, m).max(axis=(3, 5))
        self.switch = np.any([input == i for i in res.flatten()], axis=0)
        return res

    def backprop_delta(self, target):
        self.delta = self.next.backprop_delta(target).reshape((self.batch, ) + self.shape)
        # res = np.zeros(self.prev.shape)
        # res[self.switch] = self.delta.flatten()
        res = self.delta.repeat(self.pool_shape[0], axis=2)\
                        .repeat(self.pool_shape[1], axis=3)\
                        * self.switch
        return res

    def __str__(self):
        res = lm.AbstractLayer.__str__(self)
        res += '   ->   pool shape: {}'.format(self.pool_shape)
        return res
