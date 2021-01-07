import numpy as np

class Network:


    def __init__(self, network):
        self.network = network


    def forward(self, input):
        for layer in self.network:
            input = layer.forward(input)

    # def predict(self, input):
    #     """ zwraca indeks największej wartości z ostatniej warstwy"""
    #     results = self.forward(input)[-1]
    #     return np.argmax(results, axis=-1)

    def loss_function(self, results, y):
        y_vector = np.zeros(len(self.network[-1].biases))
        y_vector[y] = 1
        # TODO czy result nie powinien byc z przedzialu 0,1
        return np.sum(np.square(np.subtract(y_vector, results)))
    
    def loss_derivative(self, results, y):
        y_vector = np.zeros(len(self.network[-1].biases))
        y_vector[int(y)] = 1
        return np.dot(2, np.subtract(results, y_vector))

    def backward_propagation(self, input, y):
        self.forward(input)
        # last layer
        # last_layer = self.network[-1]

        gradient_w = [[] for i in range(len(self.network) - 1)]
        gradient_b = [[] for i in range(len(self.network) - 1)]

        # grad_a = np.array([0.0 for _ in range(len(last_layer.biases))])
        # # grad_a = 0.0
        # loss_der = np.sum(self.loss_derivative(last_layer.a, y))   # 2(a^L - y) TODO czy powinna byc sumą
        # for weights, z in zip(last_layer.weights, last_layer.z):
        #     z_der = np.sum(self.network[-2].a)  # active(L-1)
        #     a_der = last_layer.activation_derivative(z)  # activation'(z^L)
            # gradient_w[-1].append(np.dot(np.dot(z_der, a_der), loss_der))
            # gradient_b[-1].append(np.dot(a_der, loss_der))
            # grad_a = np.add(grad_a, np.dot(np.dot(weights, a_der), loss_der))
            
        for layer_num in range(1, len(self.network)):
            curr_layer = self.network[-layer_num]
            if layer_num == 1:
                loss_der = np.sum(self.loss_derivative(curr_layer.a, y))
                grad_a = np.array([0.0 for _ in range(len(curr_layer.biases))])
            else:
                loss_der = np.sum(grad_a)   # TODO na pewno np.sum i niżej np.sum?
                grad_a = np.array([loss_der for _ in range(len(curr_layer.biases))])
            for weights, z in zip(curr_layer.weights, curr_layer.z):
                z_der = np.sum(self.network[-layer_num - 1].a)
                a_der = curr_layer.activation_derivative(z)
                gradient_w[-layer_num].append(np.dot(np.dot(z_der, a_der), loss_der))
                gradient_b[-layer_num].append(np.dot(a_der, loss_der))
                grad_a = np.add(grad_a, np.dot(np.dot(weights, a_der), loss_der))
        return gradient_w, gradient_b






            

    #     activ_der = last_layer.der_active_z     # activation'(z^L)
    #     z_der = last_layer.prev_active  # active(L-1)
    #     deriv_al = np.dot(loss_der, activ_der)
    #     grad_w = np.dot(deriv_al, z_der)
    #     grad_b = deriv_al
    #     grad_prev_a = np.dot(last_layer.weights, loss_der)


    #     for layer in self.network:
    #         pass

