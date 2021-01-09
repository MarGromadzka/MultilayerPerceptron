import numpy as np
from Layers import Layer

class Network:
    def __init__(self, layers_size):
        self.layer_size = layers_size
        # tworzenie sieci dla zadanej ilości warstw o określonej wielkości
        self.network = []
        self.network.append(Layer(layers_size[0], layers_size[0], first_layer=True))
        for i in range(len(layers_size) - 2):
            self.network.append(Layer(layers_size[i], layers_size[i + 1]))
        self.network.append(Layer(layers_size[-2], layers_size[-1], last_layer=True))

    def forward(self, input):
        # input w postaci [i, i, ... i, i]
        input = np.array([input]).T
        for layer in self.network:
            input = layer.forward(input)

    def predict(self, input):
        """ zwraca indeks największej wartości z ostatniej warstwy"""
        self.forward(input)
        return np.argmax(self.network[-1].activation)

    def loss_function(self, results, y):
        ''' liczy funkcję straty
        params:
            result - wektor wyników funkcji aktywacji dla ostatniej warstwy
            y - index który z rezultatów powinien być największy
        return:
            funkcja straty o wzorze
            1/N * Σ (result - wektor składający się z 0 i jedynki)^2
        '''
        y_vector = np.zeros(len(results))
        y_vector[y] = 1.0
        y_vector = np.array([y_vector]).T
        results = results.astype(float) / (results.max() - results.min())
        results = [max(np.zeros(1),result) for result in results]
        # TODO czy result nie powinien byc z przedzialu 0,1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        return np.sum(np.subtract(results, y_vector)**2, axis=0)[0] / len(y_vector)

    def loss_derivative(self, results, y):
        ''' liczy pochodną funkcji straty
        params:
            result - wektor wyników funkcji aktywacji dla ostatniej warstwy
            y - index który z rezultatów powinien być największy
        return:
            pochodną funkcji straty o wzorze
            2 * (results - wektor składający się z 0 i jedynki)
        '''
        y_vector = np.zeros(len(self.network[-1].biases))
        y_vector[int(y)] = 1.0
        y_vector = np.array([y_vector]).T
        return np.dot(2, np.subtract(results, y_vector))

    def backward_propagation(self, y):
        ''' algorytm propagacji wstecznej
        params:
            y - index który z rezultatów powinien być największy, potrzebny przy liczeniu funkcji straty
        return:
            listę obliczonych wag i biasów dla warstw poza ostatnią
        '''
        # lista w której będziemy przechowywać wyliczone gradienty
        gradient_w = [[] for i in range(len(self.network) - 1)]
        gradient_b = [[] for i in range(len(self.network) - 1)]
        # ostatnia warstwa - brany pod uwagę koszt liczony z funkcji straty
        # 
        cost_z_der = self.network[-1].activation_derivative(self.network[-1].z) * self.loss_derivative(self.network[-1].activation, y)
        for layer_num in range(-1, -len(self.network), -1):
            gradient_w[layer_num] = np.dot(cost_z_der, np.transpose(self.network[layer_num - 1].activation))
            gradient_b[layer_num] = cost_z_der
            curr_layer = self.network[layer_num - 1]
            cost_z_der = np.dot(np.transpose(self.network[layer_num].weights), cost_z_der) * curr_layer.activation_derivative(curr_layer.z)
        return gradient_w, gradient_b

    def train_network(self, train_zip, beta=0.01, epoch=4, epsilon=0.000001, batch_size=32):
        ''' trenuje sieć neuronową o zadanej ilości warstw z wykorzystaniem stochastycznego spadku gradientu + batch
        params:
            train_zip - lista składająca się z zipowanego obrazka oraz labelki
            beta - parametr kroku uczenia się
            epoch - ilość epok
            epsilon - różnica między kolejno wyliczonymi stratami funkcji, możliwość wcześniejszego zakończenia nauki przez sieć
            batch_size - wielkość treningu po którym wagi zostaną zaktualizowane
        return:
            ---
        '''
        len_train_zip = len(train_zip)
        # pojemniki na dane zanim zostaną zaktualizowane wagi - iteracje % batch_size = 0
        gradient_w = [np.zeros((j, i))for j, i in zip(self.layer_size[1::], self.layer_size[:-1:])]
        gradient_b = [np.zeros((i, 1)) for i in self.layer_size[1::]]

        for e in range(epoch):
            old_loss = 0
            # w celu randomizacji zawartości batcha
            np.random.shuffle(train_zip)
            for index, train in enumerate(train_zip):
                x, y = train
                # przednia i wsteczna propagacja dla aktualnie rozważanego inputu
                self.forward(x)
                grads_w, grads_b = self.backward_propagation(y)
                # dodanie do pojemnika aktualnie wyliczonych gradientów
                for i in range(len(grads_w)):
                    gradient_w[i] = np.add(gradient_w[i], grads_w[i])
                    gradient_b[i] = np.add(gradient_b[i], grads_b[i])
                # jeśli zbiór został napełniony - jest równy krotności batch size to następuje aktualizacja wag i biasów
                if (index % batch_size == 0 and index != 0 or index == len_train_zip):
                    for index, layer in enumerate(self.network[1::]):
                        layer.weights = [np.subtract(weight, np.dot(beta, np.divide(grad_w, batch_size))) for weight, grad_w in zip(layer.weights, gradient_w[index])]
                        layer.biases = [np.subtract(bias, np.dot(beta, np.divide(grad_b, batch_size))) for bias, grad_b in zip(layer.biases, gradient_b[index])]
                    # wyliczenie funkcji straty po aktualizacji i porównanie jej z poprzednią
                    new_loss = (self.loss_function(self.network[-1].activation, y))
                    if (abs(new_loss - old_loss) < epsilon):
                        # print("I'm here :)")
                        return
                    if np.isinf(new_loss):  # Overflow from line 37 TODO? !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        return

                    # print(f"Epoch: {e + 1} New loss: {new_loss}")
                    # przygotowanie pojemników na nowy batch
                    old_loss = new_loss
                    gradient_w = [np.zeros((j, i))for j, i in zip(self.layer_size[1::], self.layer_size[:-1:])]
                    gradient_b = [np.zeros((i, 1)) for i in self.layer_size[1::]]
            # print(f"Epoch: {e + 1}")

    def count_accuracy(self, pictures, labels):
        ''' Liczy precyzje z jaką nauczona sieć identyfikuje obrazy
        Params:
            pictures - obrazki do przepuszczenia przez sieć
            labels - co sieć powinna rozpoznać
        Return:
            procent skuteczności sieci
        '''
        accuracy = 0
        for picture, label in zip(pictures, labels):
            prediction = self.predict(picture)
            if prediction == label:
                accuracy += 1
        per_accuracy = (accuracy / len(labels)) * 100
        return per_accuracy
