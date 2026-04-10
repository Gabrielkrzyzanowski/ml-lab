#Expanding the layers size 

import numpy as np 

inputs = np.array([[1.2, 5.1, 2.1, 4.0],
                 [2.0, 3.1, 4.2, 6.7],
                 [3.8, 0.3, 9.2, 9.7]]) 

weights = np.array([[3.1, 2.1, 4.5, 1.2], 
                    [3.0, 1.1, 4.4, 0.2], 
                    [1.5, 1.6, 6.5, 5.4]]) 

biases = np.array([3,2,8]) 


weights2 = np.array([[5.1, 6.7, 3.6], 
                    [3.6, 3.2, 7.5], 
                    [1.7, 4.3, 6.3]]) 

biases2 = np.array([5,3,7]) 

layer1_output = np.dot(inputs, weights.T) + biases 

layer2_output = np.dot(layer1_output, weights2.T) + biases2 

print(layer2_output)