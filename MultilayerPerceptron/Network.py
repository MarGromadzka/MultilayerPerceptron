import numpy as np

class Network:


    def __init__(self, network):
        self.network = network


    def forward(self, X):
        activations = []
        input = X
        for layer in self.network:
            activations.append(layer.forward(input))
            input = activations[-1]
        assert len(activations) == len(self.network)
        return activations


    def predict(self, X):
        """ zwraca indeks największej wartości z ostatniej warstwy"""
        results = self.forward(X)[-1]
        return np.argmax(results, axis=-1)


    def loss_function(self, X, y):
        results = self.forward(X)[-1]
        return (y-results)**2