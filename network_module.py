import layer_module as lm
import cPickle
from utilities import StatusBar, ensure_dir


class network(object):

    @classmethod
    def load(cls, load):
        return cPickle.load(open(load, 'rb'))

    def __init__(self, *args, **kwargs):
        self.input = lm.activation(type='input', width=kwargs['in_size'])
        self.top = self.input
        self.output = lm.output(type=kwargs['criterion'],
                                prev_layer=self.input)
        self.layerlist = [self.input, self.output]

    def __getitem__(self, index):
        return self.layerlist[index]

    def register_new_layer(self, l):
        self.output.new_last_layer(l)
        self.layerlist.insert(-1, l)
        return l

    def add_dropcon(self, **kwargs):
        self.top = lm.dropcon(prev_layer=self.top, **kwargs)
        return self.register_new_layer(self.top)

    def add_full(self, width):
        self.top = lm.fully_connected(
            width=width, prev_layer=self.top)
        return self.register_new_layer(self.top)

    def add_activation(self, type, **kwargs):
        if type == 'dropout':
            self.top = lm.dropout(prev_layer=self.top, **kwargs)
        else:
            self.top = lm.activation(type=type, prev_layer=self.top, **kwargs)

        return self.register_new_layer(self.top)

    def train(self, input_set, target_set, epoch, rate, **kwargs):
        'setting utilities'
        bar = StatusBar(epoch)
        cp_name = kwargs.get('checkpoint')

        for e in xrange(epoch):
            '''
            calculating criterion yields the FORWARD propagation
            calling the input's "backprop_delta" yields BACKWARD propagation

            During one epoch the crit of the network is averaged
            and updated in the status bar.

            No batch training is used here, meaning, that:
            the network's weights are updated after each FORWARD-BACKWARD cycle
            '''
            crit = 0
            for input, target in zip(input_set, target_set):
                'FORWARD'
                crit += self.output.get_crit(input, target)

                'BACKWARD'
                self.input.backprop_delta(target)

                curr = self.output
                while curr.prev_layer is not None:
                    'iterating over layers to train them, if trainable'
                    curr = curr.prev_layer
                    curr.train(rate)
                    '''i.e. activation layers cannot be trained because they
                       doesn't hold any changeable parameters'''

            crit /= len(input_set)
            bar.update(crit)
            'Saving checkpoints during training is useful'
            if cp_name and (e + 1) % (epoch / 2) == 0:
                'By default it is done at half, and the end of the training'
                file_name = str(cp_name)
                file_name += '-e{}-ID'.format(e + 1)
                file_name += str(id(self)) + '.dat'
                self.save_state(file_name)
        if cp_name:
            print("Saved checkpoint to: '" + file_name + "'")

    def save_state(self, file_name):
        ensure_dir(file_name)
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

    def get_output(self, input):
        return self.output.get_output(input)

    def classification_validate(self, input_set, target_set):
        T = 0.0
        for input, target in zip(input_set, target_set):
            T += 1 * self.output.get_output(input).argmax() == target.argmax()

        return T / len(target_set)
