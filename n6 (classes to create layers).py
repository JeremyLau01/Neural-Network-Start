import numpy as np

np.random.seed(0)

# X = inputs (X is the standard for input data)
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


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

'''creating new network layer object/instance
Layer_Dense(# of values in each sample in this case it is 4, any # of neurons you want)'''
layer1 = Layer_Dense(4, 5)

'''layer2's inputs are layer1's outputs
therefore, the sizes of output neurons in layer1
must match # input values in layer2 in this case 5)'''
layer2 = Layer_Dense(5, 2)

# now can pass data through the layers

# passing inputs into layer1, and this produces an output we wil use as layer2's inputs
layer1.forward(X)
#print(layer1.output)

# passing layer1's output into layer2's inputs
layer2.forward(layer1.output)
print(layer2.output)

# What you are looking at (what is printed):
# if you look at the first item in the list [0.148296, -0.08397602]
# this is the output from the first inputs from X [1, 2, 3, 2.5]

