# Simple example of operations in a network layer
inputs = [1.2, 5.1, 2.1, 4.0] 

weights = [[3.1, 2.1, 4.5, 1.2], 
          [3.0, 1.1, 4.4, 0.2], 
          [1.5, 1.6, 6.5, 5.4]] 

bias = [3,2,1]

layer_output = [] 

for neuron_weights, neuron_bias in zip(weights, bias): 
    neuron_output = 0.0 
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight  
    neuron_output += neuron_bias 
    layer_output.append(neuron_output)


print(layer_output)