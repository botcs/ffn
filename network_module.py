import layer_module as lm
import cPickle


class network(object):

    @classmethod
    def load(cls, load):
        return cPickle.load(open(load, 'rb'))

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

    def train(self, input_set, target_set, epoch, rate, **kwargs):
        cp_name = kwargs.get('checkpoint')
        for e in xrange(epoch):
            for input, target in zip(input_set, target_set):
                self.output.get_output(input)
                self.input.backprop_error(target)

                curr = self.output
                while curr.prev_layer is not None:
                    curr = curr.prev_layer
                    curr.train(rate)

            if (e + 1) % (epoch / 10) == 0:
                crit = self.output.get_crit(input, target)
                print ('ep. {} error: {}'.format(e + 1, crit))

            if (e + 1) % (epoch / 2) == 0:
                if cp_name:
                    file_name = str(cp_name)
                    file_name += '-e{}-ID'.format(e + 1) + \
                        str(id(self)) + '.dat'
                    self.save_state(file_name)
                    print("Saved checkpoint to: '" + file_name + "'")

    def save_state(self, file_name=None):
        if file_name:
            cPickle.dump(self, open(file_name, 'wb'))
        else:
            cPickle.dump(self, open(str(id(self)) + '.dat', 'wb'))

    def __str__(self):
        res = 'Network ID: ' + str(id(self))
        res += '\nNetwork layout:\n'
        res += '-' * 30 + '\n'
        res += '\tINPUT[{}]'.format(self.input.width)
        curr = self.input
        while curr is not None:
            res += ('\n\t   |\n')
            res += curr.__str__()
            curr = curr.next_layer

        res += '\n' + '-' * 30
        return res

    def classification_validate(self, input_set, target_set):
        T = 0.0
        for input, target in zip(input_set, target_set):
            T += 1 * self.output.get_output(input).argmax() == target.argmax()

        return T / len(target_set)
