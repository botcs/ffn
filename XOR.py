import network_module as nm
import numpy as np


logic_table = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
xor_label = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])
and_label = np.array([[0, 1], [0, 1], [0, 1], [1, 0]])
or_label = np.array([[0, 1], [1, 0], [1, 0], [1, 0]])

nn = nm.network(in_size=2, criterion='MSE')
nn.add_full(3)
nn.add_activation(type='tanh')
# nn.add_dropcon(width=3, p=0.3)
nn.add_full(2)

nn.input.next_layer.perturb(10)
nn.output.prev_layer.perturb(10)

nn.train(input_set=logic_table, target_set=xor_label,
         epoch=10000, rate=0.1, checkpoint='./test_dir/xor_HOT_perturb')
