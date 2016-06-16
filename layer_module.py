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

    def backprop_error(self, next_layer):
        pass


class fully_connected(_layer):

    def __init__(self, *args, **kwargs):
        _layer.__init__(self, *args, **kwargs)
        '''ROW REPRESENTS OUTPUT NEURON'''
        self.input_weights = np.random.rand(self.width, self.prev_layer.width)
        self.bias = np.random.rand(self.width)
        self.output = np.zeros([self.width, 1])

    def get_local_output(self, input):
        return np.dot(self.input_weights, input) + self.bias

    def backprop_error(self, target):
        '''error is defined as the partial derivative of the cost function
        with respect to the ***OUTPUT*** of the current layer

        '''
        if self.next_layer:
            self.error = self.next_layer.backprop_error(target)
        else:
            print('NO OUTPUT LAYER SPECIFIED')
        return np.dot(self.error, self.input_weights)


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

    def get_crit(self, target):
        return output.crit[self.type]((self.output, target))

    def backprop_error(self, target):
        self.error = output.derivative[self.type]((self.output, target))
        return self.error


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
        'atan': lambda x: np.arctan(x),
        'soft_step': lambda x: 1 / (1 + np.exp(-x)),
    }

    derivative_functions = {
        'identity': lambda x: np.ones(x.shape),
        'relu': lambda x: 1 * (x > 0),
        'atan': lambda x: 1 / (1 + x**2),
        'soft_step': lambda x: 1 / (1 + np.exp(-x)) - (1 + np.exp(-x))**-2
    }

    def act(self, x):
        return activation.activation_functions[self.type](x)

    def der(self, x):
        return activation.derivative_functions[self.type](x)

    def __init__(self, *args, **kwargs):
        if kwargs['type'] == 'input':
            self.type = 'identity'
            self.prev_layer = None
            self.width = kwargs['width']
        else:
            self.type = kwargs['type']
            self.prev_layer = kwargs.get('prev_layer')
            self.width = self.prev_layer.width

    def get_local_output(self, input):
        return self.act(input)

    def backprop_error(self, target):
        self.error = self.next_layer.backprop_error(target)
        return self.der(self.error) * self.error
