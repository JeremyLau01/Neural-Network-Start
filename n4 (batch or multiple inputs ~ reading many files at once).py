import numpy as np

# Batch of inputs
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

'''
Need to use .T (transpose) because inner dimensions of
inputs and weights need to match. Before, the size of
inputs was (4,) so np.dot compares the sizes of the last
dimensions or 'axis' of each matrix/vector:
(3, 4) (4,) before   -v.s.-   (3, 4) (4, 3) after    '''
output = np.dot(inputs, np.array(weights).T) + biases
print(output)
