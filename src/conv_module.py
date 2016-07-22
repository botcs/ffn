from scipy.signal import convolve2d
import numpy as np
import layer_module as lm

np.set_printoptions(precision=2, edgeitems=2, threshold=5)


class conv(lm._layer):

    def __init__(self, num_of_featmap, kernel_shape, **kwargs):
        lm._layer.__init__(self, **kwargs)
        self.type = 'convolution'
        self.kernel_shape = kernel_shape
        self.nof = num_of_featmap
        "prev layer's shape[0] is the number of output channels/feature maps"
        self.kernels = np.random.randn(
            self.prev.shape[0], self.nof, *self.kernel_shape)

        self.biases = np.random.randn(self.prev.shape[0], self.nof)

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
        return np.ndarray(
            [[np.sum([convolve2d(c, k, 'valid')
                      for c, k in zip(channels, kernel_set)])
              for channels, kernel_set in zip(sample, self.kernels)]
             for sample in self.input])

    def backprop_delta(self, target):
        self.delta = self.next.backprop_delta(target).reshape(self.shape)

        '''Each feature map is the result of all previous layer maps,
        therefore the same gradient has to be spread for each'''
        return np.sum([
            [[convolve2d(k[::-1, ::-1], d) for k in kernel_set]
             for d, kernel_set in zip(sample_delta, self.kernels)]
            for sample_delta in self.delta], axis=1)

    def get_param_grad(self):
        pass
    
    def acc_grad(self):
        grad = self.get_param_grad()
        try:
            self.acc_count += 1
            'Kernels'
            self.k_acc += grad[0]
            'Biases'
            self.b_acc += grad[1]
        except AttributeError:
            self.acc_count = 1
            self.k_acc, self.b_acc = grad

    def SGDtrain(self, rate):
        self.kernels -= rate / self.acc_count * self.k_acc
        self.biases -= rate / self.acc_count * self.b_acc
        self.acc_count = 0
        self.k_acc = [np.zeros(k.shape) for k in self.kernels]
        self.b_acc = np.zeros(self.biases.shape)

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
        res = input.reshape(N, h/n, n, w/m, m).max(axis=(2, 4))
        'Keep record of which neurons were chosen in the pool by their index'
        # self.switch = np.any(
        #     [input == i for i in res.flatten()], axis=0).nonzero()
        self.switch = np.any([input == i for i in res.flatten()], axis=0)
        return res

    def backprop_delta(self, target):
        self.delta = self.next.backprop_delta(target).reshape(self.shape)
        # res = np.zeros(self.prev.shape)
        # res[self.switch] = self.delta.flatten()
        res = self.delta.repeat(self.pool_shape[0], axis=1)\
                        .repeat(self.pool_shape[1], axis=2)\
                        * self.switch
        return res

    def __str__(self):
        res = lm._layer.__str__(self)
        res += '   ->   pool shape: {}'.format(self.pool_shape)
        return res
