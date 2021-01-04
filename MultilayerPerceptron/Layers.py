import numpy as np 

class Layer:

    def __init__(self, input_size, output_size, use_activation_function):
        self.weights = np.random.normal(loc = 0.0, scale = np.sqrt(2/(input_size+output_size)), size = (input_size, output_size))
        self.biases = np.zeros(output_size)
        self.use_activation_function = use_activation_function


    def forward(self, input):
        f = np.dot(input, self.weights) + self.biases
        if self.use_activation_function:
            return np.arctan(f)
        return f

