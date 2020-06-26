import numpy as np

'''From first layer to second layer'''
# Batch of inputs
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

# First matrix (2D array) of weights
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

# First list (vector or 1D array) of biases
biases = [2, 3, 0.5]

# "Batch of inputs" inputs = output of inputs*weights + biases

'''From second layer to third layer'''
# Second matrix (2D array) of weights
weights2 = [[0.1, -0.14, 0.5],
           [0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

# Second list (vector or 1D array) of biases
biases2 = [-1, 2, -0.5]

'''From first layer to second layer'''
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

'''From second layer to third layer'''
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

