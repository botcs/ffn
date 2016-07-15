import numpy as np
import utilities as util
import warnings


class _layer:

    def __init__(self, **kwargs):
        'get returns None instead of raising KeyError'
        self.type = kwargs.get('type')
        self.prev = kwargs.get('prev')
        self.next = kwargs.get('next')

        self.width = kwargs.get('width')
        self.shape = kwargs.get('shape')
        if self.prev is None:
            util.warning('No previous layer defined', self)

        'Shape states for the default output of the layer'
        if self.shape and not self.width:
            self.width = np.prod(self.shape)
        elif self.width:
            self.shape = (self.width,)

    def get_propagated_input(self, input):
        '''Recursive definition, a layer can only produce real output from the
        previous layer.

        The only case when the input is directly calculated through
        one layer, is when there is no previous layer

        '''
        if self.prev:
            input = self.prev.get_output(input)

        return input

    def get_local_output(self, input):
        pass

    def get_output(self, input):
        prop_in = self.get_propagated_input(input)
        self.output = self.get_local_output(prop_in)
        return self.output

    def backprop_delta(self, next):
        pass

    def train(self, rate):
        pass

    def __str__(self):
        res = '{}  {}'.format(self.type, self.shape)
        return res


class fully_connected(_layer):

    def __init__(self, **kwargs):
        _layer.__init__(self, type='fully connected', **kwargs)
        '''ROW REPRESENTS OUTPUT NEURON'''
        assert self.width is not None, "'shape=' or 'width=' must be defined"
        if self.prev:
            self.weights = np.random.randn(self.width, self.prev.width)
        self.bias = np.random.randn(self.width)
        self.output = np.zeros([self.width, 1])

        'Sharpening the deviation of initial values - less training required'
        self.weights /= self.width

    def perturb(self, delta):
        'strictly experimental'
        rows, cols = self.weights.shape
        self.weights += (np.random.randn(rows, cols)) * delta

    def get_local_output(self, input):
        self.input = np.array(input).flatten()
        'input will be required for training'
        return np.dot(self.weights, self.input) + self.bias

    def backprop_delta(self, target):
        if self.next:
            self.delta = self.next.backprop_delta(
                target).reshape(self.shape)
        return np.dot(self.delta, self.weights)

    def train(self, rate):
        '''GRADIENT DESCENT TRAINING'''
        self.bias -= rate * self.delta
        self.weights -= rate * np.outer(self.delta, self.input)

    def __str__(self):
        res = _layer.__str__(self)
        res += '   ->   weights + bias: {} + {}'.format(
            self.weights.shape, self.bias.shape)
        return res


class activation(_layer):
    '''
    STRICTLY ONE-TO-ONE CONNECTION, WITH f ACTIVATION FUNCTIONS

    IN1 ------ [f] ------ OUT1
    IN2 ------ [f] ------ OUT2
    IN3 ------ [f] ------ OUT3
    IN4 ------ [f] ------ OUT4
    '''
    activation_functions = {
        'identity': lambda x: x,
        'relu': lambda x: x * (x > 0),
        'tanh': lambda x: np.tanh(x),
        'atan': lambda x: np.arctan(x),
        'logistic': lambda x: 1.0 / (1.0 + np.exp(-x)),
    }

    derivative_functions = {
        'identity': lambda x: np.ones(x.shape),
        'relu': lambda x: 1.0 * (x > 0),
        'tanh': lambda x: 1.0 - np.square(np.tanh(x)),
        'atan': lambda x: 1.0 / (1.0 + x**2),
        'logistic': lambda x: activation.activation_functions['logistic'](x) *
            (1 - activation.activation_functions['logistic'](x))
    }

    def act(self, x):
        return activation.activation_functions[self.type](x)

    def der(self, x):
        return activation.derivative_functions[self.type](x)

    def __init__(self, type, **kwargs):
        if kwargs.get('prev'):
            prev_shape = kwargs.get('prev').shape
            _layer.__init__(self, shape=prev_shape, type=type, **kwargs)
        else:
            _layer.__init__(self, type=type, **kwargs)

    def __str__(self):
        return 'activation {}   ->   type: {}'.format(self.shape, self.type)

    def get_local_output(self, input):
        self.input = input
        return self.act(input)

    def backprop_delta(self, target):
        self.delta = self.next.backprop_delta(target).reshape(self.shape)
        return self.der(self.input) * self.delta


class input(activation):

    def __init__(self, shape, **kwargs):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            activation.__init__(self, shape=shape,
                                type='identity', **kwargs)


class output(activation):
    '''Special type of activation layer, that has TWO kind of output:

    FIRST is [N -> N] type, very similar to activation, but the
    activation function is not strictly one-to-one

    SECOND is [(N x N) -> 1] always, producing a
    Loss/Criterion/Cost/Error like result as a function of the FIRST
    output, and the Target

    '''

    def __init__(self, **kwargs):
        activation.__init__(self, **kwargs)

    def __str__(self):
        res = '\tOUTPUT  {}'.format(self.shape)
        res += '   ->   '
        res += 'CRITERION  ({})'.format(self.type, self.shape, 1)
        return res

    def new_last_layer(self, new_layer):
        self.prev = new_layer
        self.prev.next = self
        self.width = new_layer.width
        self.shape = new_layer.shape

    crit = {
        'MSE': lambda (prediction, target):
            0.5 * np.dot(prediction - target, prediction - target),
        'softmax': lambda (prediction, target):
            np.dot(-target, np.log(
                np.maximum(prediction, np.ones(prediction.shape) * 1e-200)))
    }

    activation = {
        'MSE': lambda x: x,
        'softmax': lambda x: np.exp(x) / np.sum(np.exp(x))
    }

    derivative = {
        'MSE': lambda (prediction, target): prediction - target,
        'softmax': lambda (prediction, target): prediction - target
    }

    '''
    for later expansion of these functions
    https://github.com/torch/nn/blob/master/doc/criterion.md
    '''

    def get_local_output(self, input):
        return output.activation[self.type](input)

    def get_crit(self, input, target):
        'double paren for lambda wrap'
        self.get_output(input)
        return output.crit[self.type]((self.output, target))

    def backprop_delta(self, target):
        '''The delta of the output layer wouldn't be used for training
        so the function returns directly the delta of the previous layer
        '''
        self.prev_delta = output.derivative[self.type]((self.output, target))
        return self.prev_delta


class dropout(activation):

    def __init__(self, p, **kwargs):
        self.type = 'dropout'
        activation.__init__(self, **kwargs)
        self.p = p

    def get_local_output(self, input):
        self.curr_drop = np.random.rand(input.size) > self.p
        return input * self.curr_drop

    def backprop_delta(self, target):
        self.delta = self.next.backprop_delta(target).reshape(self.shape)
        return self.curr_drop * self.delta


class dropcon(fully_connected):

    def __init__(self, p, **kwargs):
        fully_connected.__init__(self, **kwargs)
        self.type = 'dropcon'
        self.p = p

    def get_local_output(self, input):
        self.curr_drop = np.random.random(self.weights.shape) > self.p
        self.input = input
        return np.dot((self.curr_drop * self.weights), input)

    def backprop_delta(self, target):
        if self.next:
            self.delta = self.next.backprop_delta(target).reshape(self.delta)
        return np.dot(self.delta, self.curr_drop * self.weights)
