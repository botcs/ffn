%matplotlib inline
import matplotlib.pyplot as plt
import network_module as nm
def wplot(activations):
	for i, w in enumerate(activations):
	    plt.subplot(1, 10, i + 1)
	    plt.set_cmap('gray')
	    plt.axis('off')
	    plt.imshow(w)
	plt.gcf().set_size_inches(9, 9)
	plt.show()


