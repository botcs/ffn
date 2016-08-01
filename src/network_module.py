import layer_module as lm
import conv_module as cm
import numpy as np
import cPickle
from utilities import StatusBar, ensure_dir


class network(object):
    '''Layer manager object

    despite layers can be handled as separate instances, adding new
    components, registering them, and training becomes inefficient for
    deep-network architects

    '''

    '''Network modification definitions'''
    def pop_top_layer(self):
        assert len(self.layerlist) > 2, \
            'No hidden layers to be popped'
        popped = self.top
        # using remove, pop would remove the OUTPUT layer which is forbidden
        self.layerlist.remove(self.top)
        self.top = self.top.prev
        self.output.new_last_layer(self.top)
        return popped

    def register_new_layer(self, l):
        self.top.next = l
        l.prev = self.top
        self.top = l
        self.output.new_last_layer(l)
        self.layerlist.insert(-1, l)
        return l

    def add_conv(self, num_of_ker, kernel_shape,  **kwargs):
        self.register_new_layer(
            cm.Conv(num_of_ker, kernel_shape, prev=self.top, **kwargs))
        return self

    def add_maxpool(self, pool_shape=None, shape=None, **kwargs):
        self.register_new_layer(
            cm.max_pool(pool_shape, shape, prev=self.top, **kwargs))
        return self

    def add_dropcon(self, p, **kwargs):
        self.register_new_layer(lm.dropcon(p, prev=self.top, **kwargs))
        return self

    def add_full(self, width, **kwargs):
        self.register_new_layer(
            lm.fully_connected(width=width, prev=self.top, **kwargs))
        return self

    def add_dropout(self, p, **kwargs):
        self.register_new_layer(lm.dropout(p, prev=self.top, **kwargs))
        return self

    def add_activation(self, type, **kwargs):
        self.register_new_layer(
            lm.activation(type, prev=self.top, **kwargs))
        return self

    def add_shaper(self, shape, **kwargs):
        self.register_new_layer(
            lm.shaper(shape, prev=self.top, **kwargs))
        return self

    '''Network supervised training'''
    def train(self, input_set, target_set, epoch, rate, **kwargs):
        'setting utilities'
        bar = StatusBar(epoch)
        cp_name = kwargs.get('checkpoint')

        'For eliminating native lists, and tuples'
        input_set = np.array(input_set)
        target_set = np.array(target_set)

        result = []
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
                while curr.prev is not None:
                    'iterating over layers to train them, if trainable'
                    curr = curr.prev
                    curr.train(rate)
                    '''i.e. activation layers cannot be trained because they
                       doesn't hold any changeable parameters'''

            crit /= len(input_set)
            result.append((crit, self.test_eval(kwargs.get("test_set"))))
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
        return result

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
        '--------------------------------'
        res += '\tINPUT  {}'.format(self.input.shape)
        for i, l in enumerate(self.layerlist[1:], start=1):
            res += 2 * ('\n\t   |') + '\n\t  |{}|'.format(i) + '\n  '
            res += l.__str__()

        res += '\n' + '-' * 30
        return res

    @classmethod
    def load(cls, load):
        return cPickle.load(open(load, 'rb'))

    def __init__(self, in_shape, criterion, **kwargs):
        self.input = lm.input(in_shape)
        self.top = self.input
        self.output = lm.output(type=criterion, prev=self.top)
        self.layerlist = [self.input, self.output]

    def __getitem__(self, index):
        return self.layerlist[index]

    def __repr__(self):
        return self.__str__()

    def get_output(self, input):
        return self.output.get_output(input)

    def perc_eval(self, test_set):
        T = self.test_eval(test_set)
        return T * 100.0 / len(test_set)

    def test_eval(self, test_set):
        return np.count_nonzero(self.get_output(test_set[0]).argmax(axis=1) ==
                                test_set[1].argmax(axis=1))

    'NETWORK TRAINING METHODS'
    def SGD(self, train_policy, training_set,
            batch, rate,
            validation_set=None, epoch_call_back=None, **kwargs):

        for l in self.layerlist:
            'Set the training method for layers where it is implemented'
            try:
                l.train = l.SGDtrain
            except AttributeError:
                continue

        'For eliminating native lists, and tuples'
        input_set = np.array(training_set[0])
        target_set = np.array(training_set[1])
        
        assert len(input_set) == len(target_set),\
            'input and training set is not equal in size'
        while train_policy(training_set, validation_set, **kwargs):
            num_of_batches = len(input_set)/batch
            for b in xrange(num_of_batches):
                'FORWARD'
                self.get_output(input_set[b::num_of_batches])

                'BACKWARD'
                self.input.backprop_delta(target_set[b::num_of_batches])

                'PARAMETER GRADIENT ACCUMULATION'
                for l in self.layerlist:
                    l.train(rate)

            if epoch_call_back:
                'Some logging function is called here'
                epoch_call_back()

    'NETWORK TRAINING POLICIES'
    def fix_epoch(self, training_set, validation_set, **kwargs):
        try:
            self.last_epoch += 1
            return self.last_epoch <= kwargs['epoch']
        except AttributeError:
            self.last_epoch = 1
            return True

    def fix_hit_rate(self, training_set, validation_set, **kwargs):
        return self.perc_eval(validation_set) < kwargs['valid']

    def stop_when_overfit(self, training_set, validation_set, **kwargs):
        try:
            prev_hit = self.last_hit
            self.last_hit = self.test_eval(validation_set)
            return prev_hit < self.last_hit
        except AttributeError:
            self.last_hit = self.test_eval(validation_set)
