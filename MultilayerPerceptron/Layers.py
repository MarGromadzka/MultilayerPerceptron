import numpy as np 

class Layer:

    def __init__(self, input_size, output_size, first_layer=False, last_layer=False):
        self.last_layer_property = last_layer
        self.weights = np.random.uniform(-1 / np.sqrt(input_size), 1 / np.sqrt(input_size), (output_size, input_size))
        if first_layer:
            self.weights = np.ones((output_size, input_size))
        self.biases = np.zeros(output_size)
        self.z = None
        self.a = None

    def activation_funtion(self, x):
        if self.last_layer_property:
            return x
        return np.arctan(x)

    def activation_derivative(self, x):
        if self.last_layer_property:
            return np.ones(np.shape(x))
        return np.divide(1, np.add(1, np.square(x)))

    def forward(self, input):
        self.z = np.dot(self.weights, input) + self.biases
        self.a = self.activation_funtion(self.z)
        return self.a
