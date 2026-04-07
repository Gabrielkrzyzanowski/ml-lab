#Same example as p1 but now with numpy 

import numpy as np 

input = np.array([1.2, 5.1, 2.1, 4.0]) 

weights = np.array([[3.1, 2.1, 4.5, 1.2], 
                    [3.0, 1.1, 4.4, 0.2], 
                    [1.5, 1.6, 6.5, 5.4]]) 

bias = np.array([3,2,1]) 

layer_output = np.dot(weights,input) + bias 

print(layer_output)