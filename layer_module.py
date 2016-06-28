import numpy as np


class _layer:

    def __init__(self, *args, **kwargs):
        'get returns None instead of raising KeyError'
        self.type = kwargs.get('type')
        self.prev_layer = kwargs.get('prev_layer')
        self.width = kwargs.get('width')
        self.next_layer = kwargs.get('next_layer')

        if self.prev_layer:
            self.prev_layer.next_layer = self

    def get_propagated_input(self, input):
        '''Recursive definition, a layer can only produce real output from the
        previous layer.

        The only case when the input is directly calculated through
        one layer, is when there is no previous layer

        '''
        if self.prev_layer:
            input = self.prev_layer.get_output(input)

        return input

    def get_local_output(self, input):
        pass

    def get_output(self, input):
        self.output = self.get_local_output(self.get_propagated_input(input))
        return self.output

    def backprop_delta(self, next_layer):
        pass

    def train(self, rate):
        pass


class fully_connected(_layer):

    def __init__(self, *args, **kwargs):
        _layer.__init__(self, *args, **kwargs)
        self.type = 'fulcon'
        '''ROW REPRESENTS OUTPUT NEURON'''
        self.weights = np.random.rand(self.width, self.prev_layer.width)
        self.bias = np.random.rand(self.width)
        self.output = np.zeros([self.width, 1])

    def __str__(self):
        res = '[{} -> {}] {}:'.format(self.prev_layer.width,
                                      self.width, self.type)

        return res

    def perturb(self, delta):
        'strictly experimental'
        self.weights += (np.random.random(self.weights.shape) - 0.5) * delta

    def get_local_output(self, input):
        self.input = input
        'input will be required to train'
        return np.dot(self.weights, input) + self.bias

    def backprop_delta(self, target):
        '''delta is defined as the partial derivative of the cost function
        with respect to the ***OUTPUT*** of the current layer

        '''
        if self.next_layer:
            self.delta = self.next_layer.backprop_delta(target)
        else:
            print('NO OUTPUT LAYER SPECIFIED')
        return np.dot(self.delta, self.weights)

    def train(self, rate):
        '''GRADIENT DESCENT TRAINING'''
        self.bias -= rate * self.delta

        self.weights -= rate * np.outer(self.delta, self.input)


class output(_layer):
    '''Special type of activation layer, that has TWO kind of output:

    FIRST is [N -> N] type, very similar to activation, but the
    activation function is not strictly one-to-one

    SECOND is [(N x N) -> 1] always, producing a
    Loss/Criterion/Cost/Error like result as a function of the FIRST
    output, and the Target

    '''

    def __init__(self, *args, **kwargs):
        _layer.__init__(self, *args, **kwargs)

    def __str__(self):
        res = '[{} -> {}] OUTPUT: {}\n'.format(
            self.width, self.width, self.type)
        res += '\t   |\n'
        res += '[{} -> {}] CRITERION: {}'.format(self.width, 1, self.type)
        return res

    def new_last_layer(self, new_layer):
        self.prev_layer = new_layer
        self.prev_layer.next_layer = self
        self.width = new_layer.width

    crit = {
        'MSE': lambda (prediction, target):
            0.5 * np.dot(prediction - target, prediction - target),
        'softmax': lambda (prediction, target):
            np.dot(target, np.log(prediction)) * -1
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

    'double paren for lambda wrap'

    def get_crit(self, input, target):
        return output.crit[self.type]((self.get_output(input), target))

    def backprop_delta(self, target):
        self.delta = output.derivative[self.type]((self.output, target))
        return self.delta


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
        'logistic': lambda x: 1 / (1 + np.exp(-x)),
    }

    derivative_functions = {
        'identity': lambda x: np.ones(x.shape),
        'relu': lambda x: 1 * (x > 0),
        'tanh': lambda x: 1 - np.square(np.tanh(x)),
        'atan': lambda x: 1 / (1 + x**2),
        'logistic': lambda x: 1 / (1 + np.exp(-x)) - 1 / np.square(1 + np.exp(-x))
    }

    def act(self, x):
        return activation.activation_functions[self.type](x)

    def der(self, x):
        return activation.derivative_functions[self.type](x)

    def __init__(self, *args, **kwargs):
        _layer.__init__(self, *args, **kwargs)
        if kwargs.get('type') == 'input':
            self.type = 'identity'
            self.prev_layer = None
        else:
            self.width = self.prev_layer.width

    def __str__(self):
        res = '[{} -> {}] activation: {}'.format(self.width,
                                                 self.width,
                                                 self.type)
        return res

    def get_local_output(self, input):
        return self.act(input)

    def backprop_delta(self, target):
        self.delta = self.next_layer.backprop_delta(target)
        return self.der(self.delta) * self.delta


class dropout(activation):

    def __init__(self, *args, **kwargs):
        self.type = 'dropout'
        activation.__init__(self, *args, **kwargs)
        self.p = kwargs['p']

    def get_local_output(self, input):
        self.curr_drop = np.random.rand(input.size) > self.p
        return input * self.curr_drop

    def backprop_delta(self, target):
        self.delta = self.next_layer.backprop_delta(target)
        return self.curr_drop * self.delta


class dropcon(fully_connected):

    def __init__(self, *args, **kwargs):
        fully_connected.__init__(self, *args, **kwargs)
        self.type = 'dropcon'
        self.p = kwargs['p']

    def get_local_output(self, input):
        self.curr_drop = np.random.random(self.weights.shape) > self.p
        self.input = input
        return np.dot((self.curr_drop * self.weights), input)

    def backprop_delta(self, target):
        if self.next_layer:
            self.delta = self.next_layer.backprop_delta(target)
        else:
            print('NO OUTPUT LAYER SPECIFIED')
        return np.dot(self.delta, self.curr_drop * self.weights)
