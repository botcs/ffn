import numpy as np
import pdb
from mnist import MNIST

mndata = MNIST('.')
img, lbl = mndata.load_training()


class network:

    def act_function(self, x):
        '''should be modified with act_function_diff'''
        return x

    def act_diff(self, x):
        return 1

    def cost(self, x, y):
        '''y is the target, x is the prediction'''
        return np.dot(x - y, x - y) / 2

    def cost_diff(self, x, y):
        '''dC/dx = (x-y)'''
        return x - y

    def propagate(self, input_sample):
        result = np.array(input_sample)
        for layer in self.layers:
            layer.weighted_input = np.dot(result, layer.input_weights)
            layer.activation = layer.weighted_input + layer.bias
            result = layer.activation

        return result

    def get_grad(self, in_sample, target):
        prediction = self.propagate(in_sample)
        diff_b = [np.zeros(l.bias.shape) for l in self.layers]
        diff_w = [np.zeros(l.input_weights.shape) for l in self.layers]

        '''delta is the difference between the "correct" weighted input of the
        layer (before the activation function)'''
        delta = np.vectorize(self.cost_diff)(prediction, target) \
            * self.act_diff(self.layers[-1].weighted_input)

        '''Noob algebra:
        if A.shape = (n,) and B.shape = (m,) then they are both one-dimensional
        if n=m then an inner their inner product can be defined.

        For nasty matrix algebra:
        Two one dimensional vector (n,1) and
        (m,1) product is an inner product <A,B> = A'.B, where ' is the
        transpose operator

        But as usual matrix dot product: <(x,y),(y,z)>=(x,z)
        <B,A> will be an (n,m) matrix
        '''
        delta.shape += (1,)
        for l in self.layers:
            l.activation.shape += (1,)

        diff_b[-1] = delta
        diff_w[-1] = np.dot(self.layers[-2].activation, delta.transpose())

        for i in xrange(2, len(self.layers)):
            delta = np.dot(self.layers[-i + 1].input_weights, delta)\
                * np.vectorize(self.act_diff)(self.layers[-i])
            diff_b[-i] = delta

            diff_w[-i] = np.dot(
                self.layers[-i - 1].activation, delta.transpose())

        for b in diff_b:
            b = b.ravel()
        return (diff_b, diff_w)

    def test_and_train(self, trainset, rho):
        '''Evaluate partial error for each input X, and take the corrections'
        mean.'''
        delta_b = [np.zeros(layer.bias.shape)
                   for layer in self.layers]
        delta_w = [np.zeros(layer.input_weights.shape)
                   for layer in self.layers]

        for input, target in trainset:
            'Simply sum the partial delta weight/bias'
            partial_db, partial_dw = self.get_grad(input, target)
            delta_b = [db.ravel() + p_db.ravel()
                       for db, p_db in zip(delta_b, partial_db)]
            delta_w = [dw + p_dw for dw, p_dw in zip(delta_w, partial_dw)]

        '''Gradient descent method'''
        for b, db in zip((l.bias for l in self.layers), delta_b):
            b -= rho * db / len(trainset)
        for w, dw in zip((l.input_weights for l in self.layers), delta_w):
            w -= rho * dw / len(trainset)

    class layer:

        def __init__(self, prev_width=None, width=None):
            self.width = width
            self.input_weights = np.random.rand(
                prev_width, self.width)

            self.weighted_input = np.zeros([self.width, 1])
            self.bias = np.random.rand(self.width)
            'act_function(weighted_input + bias) = activation'
            self.activation = np.zeros([self.width, 1])

    def __init__(self, width_list=None, *args, **kwargs):
        if width_list:
            self.layers = []
            for i, w in enumerate(width_list):
                if i == 0:
                    continue
                self.layers.append(self.layer(width_list[i - 1], w))

        else:
            self.input_w = kwargs['input_w']
            self.hidden_w = kwargs['hidden_w']
            self.output_w = kwargs['output_w']
            self.hidden_count = kwargs['hidden_count']

            self.layers = [self.layer(
                prev_width=self.input_w, width=self.hidden_w)]

            self.layers.extend([self.layer(
                width=self.hidden_w, prev_width=self.hidden_w)
                for i in xrange(self.hidden_count - 1)])

            self.layers.append(self.layer(
                prev_width=self.hidden_w, width=self.output_w))

    def __str__(self):
        for layer in self.layers:
            print layer.input_weights.shape
        return 'weight matrix shapes'
