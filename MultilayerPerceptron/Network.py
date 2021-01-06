import numpy as np

class Network:


    def __init__(self, network):
        self.network = network


    def forward(self, input):
        activations = []
        for layer in self.network:
            activations.append(layer.forward(input))
            input = activations[-1]
        assert len(activations) == len(self.network)
        return activations


    def predict(self, input):
        """ zwraca indeks największej wartości z ostatniej warstwy"""
        results = self.forward(input)[-1]
        return np.argmax(results, axis=-1)


    def loss_function(self, input, y):
        y_vector = np.zeros(len(self.network[-1].biases))
        y_vector[y] = 1;
        results = self.forward(input)[-1]
        return np.sum((y_vector-results)**2)