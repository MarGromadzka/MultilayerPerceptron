import numpy as np 

class Layer:

    def __init__(self, input_size, output_size, use_activation_function):
        self.weights = np.random.normal(loc = 0.0, scale = np.sqrt(2/(input_size+output_size)), size = (input_size, output_size))
        self.biases = np.zeros(output_size) # rozkład normalny
        self.use_activation_function = use_activation_function
        self.z = None
        self.a = None
        self.gradient_w = None
        self.gradient_b = None

    def activation_funtion(self, x):
        return np.arctan(x)

    def activation_derivative(self, x):
        return np.divide(1, np.add(1, np.square(x)))

    def forward(self, input):
        self.z = np.dot(input, self.weights) + self.biases
        self.a = self.activation_funtion(self.z)
        if self.use_activation_function:
            return self.a
        return self.z

    #     self.der_active_z = None  # result of decision function for backpropagation
    #     self.prev_active = None     # result of activation function for backpropagation

    # def activation_funtion(self, x):
    #     return np.arctan(x)
    
    # def activation_derivative(self, x):
    #     return 1 / (1 + x**2)

    # def forward(self, input):
    #     self.der_active_z = self.activation_derivative(input)
    #     self.prev_active = input
    #     result_z = np.dot(input, self.weights) + self.biases
    #     if self.use_activation_function:
    #         return self.activation_funtion(result_z)
    #     return result_z
