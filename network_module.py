import layer_module as lm
import cPickle
import sys


class StatusBar:

    def __init__(self, total, barLength=30):
        self.total = total
        self.curr = 0
        self.percentage = 0
        self.barLength = barLength

    def barStr(self):
        currBar = self.barLength * self.percentage / 100
        return '[' + "=" * currBar + " " * (self.barLength - currBar) + ']'

    def printBar(self, msg):
        if(self.percentage <= 100):
            print("\r  " + self.barStr() + " (" +
                  str(self.curr) + '/' + str(self.total) + ")   " +
                  str(100 * self.curr / self.total) +
                  "%  {}   ".format(msg)),
            sys.stdout.flush()
            if(self.percentage == 100):
                print '\n'

    def update(self, msg):
        self.curr += 1
        currPercentage = self.curr * 100 / self.total
        if(currPercentage > self.percentage):
            self.percentage = currPercentage
            self.printBar(msg)


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

    def add_activation(self, type, **kwargs):
        if type == 'dropout':
            self.top_layer = lm.dropout(
                type=type, prev_layer=self.top_layer, **kwargs)

        else:
            self.top_layer = lm.activation(
                type=type, prev_layer=self.top_layer, **kwargs)

        self.output.new_last_layer(self.top_layer)

    def train(self, input_set, target_set, epoch, rate, **kwargs):
        bar = StatusBar(epoch)
        cp_name = kwargs.get('checkpoint')
        for e in xrange(epoch):
            crit = 0
            for input, target in zip(input_set, target_set):
                crit += self.output.get_crit(input, target)
                self.input.backprop_delta(target)

                curr = self.output
                while curr.prev_layer is not None:
                    curr = curr.prev_layer
                    curr.train(rate)
            crit /= len(input_set)
            bar.update(crit)

            # if (e + 1) % (epoch / 10) == 0:
            #     print ('ep. {} error: {}'.format(e + 1, crit))

            if cp_name:
                if (e + 1) % (epoch / 2) == 0:
                    file_name = str(cp_name)
                    file_name += '-e{}-ID'.format(e + 1)
                    file_name += str(id(self)) + '.dat'
                    self.save_state(file_name)
                    # print("Saved checkpoint to: '" + file_name + "'")

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
