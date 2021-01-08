import numpy as np
from Layers import Layer

class Network:
    def __init__(self, layers_size):
        self.layer_size = layers_size
        self.network = []
        self.network.append(Layer(layers_size[0], layers_size[0], first_layer=True))
        for i in range(len(layers_size) - 2):
            self.network.append(Layer(layers_size[i], layers_size[i + 1]))
        self.network.append(Layer(layers_size[-2], layers_size[-1], last_layer=True))
    # TODO skalowanie na przedział 0, 1

    def forward(self, input):
        for layer in self.network:
            input = layer.forward(input)

    # def predict(self, input):
    #     """ zwraca indeks największej wartości z ostatniej warstwy"""
    #     results = self.forward(input)[-1]
    #     return np.argmax(results, axis=-1)

    def loss_function(self, results, y):
        y_vector = np.zeros(len(self.network[-1].biases))
        y_vector[y] = 1.0
        # TODO czy result nie powinien byc z przedzialu 0,1
        return np.sum(np.square(np.subtract(y_vector, results)))

    def loss_derivative(self, results, y):
        y_vector = np.zeros(len(self.network[-1].biases))
        y_vector[int(y)] = 1.0
        return np.dot(2, np.subtract(results, y_vector))

    def backward_propagation(self, input, y):
        self.forward(input)
        gradient_w = [[] for i in range(len(self.network) - 1)]
        gradient_b = [[] for i in range(len(self.network) - 1)]
        for layer_num in range(-1, -len(self.network), -1):
            curr_layer = self.network[layer_num]
            if layer_num == -1:
                cost_z_der = self.loss_derivative(curr_layer.a, y) * curr_layer.activation_derivative(curr_layer.z)   # ∂C0/∂z = ∂aL/∂zL * ∂C0/∂aL
                gradient_w[layer_num] = np.outer(cost_z_der, self.network[layer_num - 1].a)  # ∂C0/∂wL = ∂zL/∂wL * ∂C0/∂z = a(L-1)*activ(zL)*2(aL - y)
                gradient_b[layer_num] = cost_z_der       # ∂C0/∂bL = 1 * ∂C0/∂z
            else:
                prev_layer = self.network[layer_num + 1]
                cost_z_der = np.dot(np.transpose(prev_layer.weights), cost_z_der) * curr_layer.activation_derivative(curr_layer.z)
                gradient_w[layer_num] = np.outer(cost_z_der, np.transpose(self.network[layer_num - 1].a))  # ∂C0/∂wL = ∂zL/∂wL * ∂C0/∂z = a(L-1)*activ(zL)*2(aL - y)
                gradient_b[layer_num] = cost_z_der
        return gradient_w, gradient_b

    def train_network(self, train_x, train_y, beta, epoch):
        for e in range(epoch):
            for x, y in zip(train_x, train_y):
                grads_w, grads_b = self.backward_propagation(x, y)
                for index, layer in enumerate(self.network[1::]):
                    layer.weights = [weight - beta * grad_w for weight, grad_w in zip(layer.weights, grads_w[index])]
                    layer.biases = [bias - beta * grad_b for bias, grad_b in zip(layer.biases, grads_b[index])]
            print(f"{e}")