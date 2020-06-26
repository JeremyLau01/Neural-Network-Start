import numpy as np       #22:00 in part #5
import nnfs
from nnfs.datasets import spiral_data # importing sprial data

#don't have to do this, can use what had in same line in p6 - don't import nnfs also
nnfs.init() #this is from his github/youtube channel
# it sets the random seed, sets the default datatype for numpy to use, has the dataset also

# importing data, 100 feature sets of 3 classes (3 different colors in spiral, go to
# https://www.youtube.com/watch?v=gmjzbpSVY1A&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=5 for more info)
X, y = spiral_data(100, 3)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        '''.randn(sample size of 4, number of neurons)
        b/c e/a neuron has #weights = #inputs, coding for layer
        .randn allows you to define the shape of the array
        .randn generates random values using gaussian aka normal distribution bounded around 0
        all random numbers gaussian distribution with shape of array
        multiply by 0.1 because want all output-input values to be small 
        after many iterations large values increase exponentially if not multiply by 0.1
        Don't have to transpose weight matrix if initialize weights by dimensions of inputs and neurons '''
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)

        ''' initializing all biases to zero, shape of array is 1 by number of neurons
        because each neuron has one bias and coding for one layer
        if all final outputs are zero, then initialize biases to non-zero value
        First parameter of np.zeroes is a shape, so the shape is a tuple (1, n_neurons)'''
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        # outputting weighted inputs with added bias - one forward pass
        self.output = np.dot(inputs, self.weights) + self.biases

# activation ReLu function
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

'''creating new network layer object/instance
Layer_Dense(# of values in each sample here it is 2 b/c xy points on a graph, any # of neurons you want)'''
layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU() # creating activation ReLU output

layer1.forward(X)

activation1.forward(layer1.output)
print(activation1.output)
