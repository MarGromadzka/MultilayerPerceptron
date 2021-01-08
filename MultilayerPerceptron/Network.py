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
        input = np.array([input]).T
        for layer in self.network:
            input = layer.forward(input)

    def predict(self, input):
        """ zwraca indeks największej wartości z ostatniej warstwy"""
        self.forward(input)
        return np.argmax(self.network[-1].a)
        # return self.network[-1].a

    def loss_function(self, results, y):
        y_vector = np.zeros(len(self.network[-1].biases))
        y_vector[y] = 1.0
        y_vector = np.array([y_vector]).T
        # TODO czy result nie powinien byc z przedzialu 0,1
        return np.sum(np.subtract(results, y_vector)**2, axis=0)[0] / len(y_vector)

    def loss_derivative(self, results, y):
        y_vector = np.zeros(len(self.network[-1].biases))
        y_vector[int(y)] = 1.0
        y_vector = np.array([y_vector]).T
        return np.dot(2, np.subtract(results, y_vector))

    def backward_propagation(self, y):
        gradient_w = [[] for i in range(len(self.network) - 1)]
        gradient_b = [[] for i in range(len(self.network) - 1)]
        for layer_num in range(-1, -len(self.network), -1):
            curr_layer = self.network[layer_num]
            if layer_num == -1:
                cost_z_der = curr_layer.activation_derivative(curr_layer.z) * self.loss_derivative(curr_layer.a, y)
                gradient_w[layer_num] = np.dot(cost_z_der, np.transpose(self.network[layer_num - 1].a))
                gradient_b[layer_num] = cost_z_der
            else:
                prev_layer = self.network[layer_num + 1]
                cost_z_der = np.dot(np.transpose(prev_layer.weights), cost_z_der) * curr_layer.activation_derivative(curr_layer.z)
                gradient_w[layer_num] = np.dot(cost_z_der, np.transpose(self.network[layer_num - 1].a))
                gradient_b[layer_num] = cost_z_der
        return gradient_w, gradient_b

    def train_network(self, train_x, train_y, beta, epoch, epsilon, batch_size):
        train_zip = list(zip(train_x, train_y))
        len_train_zip = len(train_zip)
        gradient_w = [np.zeros((j, i))for j, i in zip(self.layer_size[1::], self.layer_size[:-1:])]
        gradient_b = [np.zeros((i,)) for i in self.layer_size[1::]]
        for e in range(epoch):
            old_loss = 0
            np.random.shuffle(train_zip)
            for index, train in enumerate(train_zip):
                x, y = train
                self.forward(x)
                grads_w, grads_b = self.backward_propagation(y)
                for i in range(len(grads_w)):
                    gradient_w[i] = np.add(gradient_w[i], grads_w[i])
                    gradient_b[i] = np.add(gradient_b[i], grads_b[i])
                if (index % batch_size == 0 and index != 0 or index == len_train_zip):
                    for index, layer in enumerate(self.network[1::]):
                        layer.weights = [np.subtract(weight, np.dot(beta, grad_w)) for weight, grad_w in zip(layer.weights, grads_w[index])]
                        layer.biases = [np.subtract(bias, np.dot(beta, grad_b)) for bias, grad_b in zip(layer.biases, grads_b[index])]
                    new_loss = (self.loss_function(self.network[-1].a, y))
                    if (abs(new_loss - old_loss) < epsilon):
                        print("I'm here :)")
                        return
                    print(f"Epoch: {e + 1} New loss: {new_loss}")
                    old_loss = new_loss
                    gradient_w = [np.zeros((j, i))for j, i in zip(self.layer_size[1::], self.layer_size[:-1:])]
                    gradient_b = [np.zeros((i,)) for i in self.layer_size[1::]]

            print(f"Epoch: {e + 1}")

    def count_accuracy(self, pictures, labels):
        accuracy = 0
        for picture, label in zip(pictures, labels):
            prediction = self.predict(picture)
            if prediction == label:
                accuracy += 1
        print(f"{(accuracy / len(labels))*100} %")
