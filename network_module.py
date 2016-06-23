import layer_module as lm


class network(object):

    def __init__(self, *args, **kwargs):
        self.input = lm.activation(type='input', width=kwargs['in_size'])
        self.top_layer = self.input

        self.output = lm.output(type=kwargs['criterion'],
                                prev_layer=self.top_layer)

    def add_full(self, width):
        self.top_layer = lm.fully_connected(
            width=width, prev_layer=self.top_layer)
        self.output.new_last_layer(self.top_layer)

    def add_activation(self, type):
        self.top_layer = lm.activation(type=type, prev_layer=self.top_layer)
        self.output.new_last_layer(self.top_layer)

    def train(self, input_set, target_set, epoch, rate):
        for e in xrange(epoch):
            for input, target in zip(input_set, target_set):
                self.output.get_output(input)
                self.input.backprop_error(target)

                curr = self.output
                while curr.prev_layer is not None:
                    curr = curr.prev_layer
                    curr.train(rate)

            if e % (epoch / 5) == 1:
                print self.output.get_crit(input, target)

    def classification_validate(self, input_set, target_set):
        T = 0.0
        for input, target in zip(input_set, target_set):
            T += 1 * self.output.get_output(input).argmax() == target.argmax()

        return T / len(target_set)
