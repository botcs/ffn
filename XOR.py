import network_module as nm
import numpy as np


logic_table = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
xor_label = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])
and_label = np.array([[0, 1], [0, 1], [0, 1], [1, 0]])
or_label = np.array([[0, 1], [1, 0], [1, 0], [1, 0]])

nn = nm.network(in_size=2, criterion='softmax')
nn.add_full(2)


nn.train(input_set=logic_table, target_set=or_label,
         epoch=10000, rate=0.05)
