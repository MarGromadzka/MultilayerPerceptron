import numpy as np 

class Layer:

    def __init__(self, input_size, output_size, first_layer=False, last_layer=False):
        ''' klasa przechowująca dane dotyczące pojedyńczego Layera
        params:
            input_size - rozmiar danych które wejdą do warstwy
            output_size - ilość neuronów
        default_params:
            first_layer - konieczny podczas realizowania wymagania na przechowywanie wartości początkowych przez pierwszą warstwę sieci
            last_layer - konieczny podczas realizowania wymagania na liniową funkcję aktywacji ostatniej warstwy
        '''
        self.last_layer_property = last_layer
        self.first_layer_property = first_layer
        self.weights = np.random.uniform(-1 / np.sqrt(input_size), 1 / np.sqrt(input_size), (output_size, input_size))
        if first_layer:
            self.weights = None
        self.biases = np.array([np.zeros(output_size)]).T
        #
        self.z = None
        # obliczona aktywacja
        self.activation = None

    def activation_funtion(self, x):
        if self.last_layer_property:
            return x
        return np.arctan(x)

    def activation_derivative(self, x):
        if self.last_layer_property:
            return np.ones(np.shape(x))
        return np.divide(1, np.add(1, np.square(x)))

    def forward(self, input):
        if self.first_layer_property:
            self.z = input
            self.activation = input
        else:
            self.z = np.sum([np.dot(self.weights, input), self.biases], axis=0)
            self.activation = self.activation_funtion(self.z)
        return self.activation
