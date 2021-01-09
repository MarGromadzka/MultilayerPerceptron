import numpy as np
from mlxtend.data import loadlocal_mnist
from Layers import Layer
from data_preparation import get_test_data, get_train_validation_data
from Network import Network

def get_best_layer_size(layer_sizes, data_zip, beta, num_epoch, epsilon, batch_size):
    ''' wyznacza najlepszy rozkład warstw dla zadanego zbioru
    params:
        layer_sizes - lista list z wielkościami warstw
        data_zip - lista składająca się z listy zipowanych wartości treningowych oraz listy niezipowanych wartości walidacyjnych
        beta - parametr kroku uczenia się
        num_epoch - ilość epok
        epsilon - różnica między kolejno wyliczonymi stratami funkcji, możliwość wcześniejszego zakończenia nauki przez sieć
        batch_size - wielkość treningu po którym wagi zostaną zaktualizowane
    return:
        nauczoną sieć neuronową na zbiorze o największej sktuczeności rozpoznawania
    '''
    train_zip, valid_set = data_zip
    rated = {}
    for layer_size in layer_sizes:
        network = Network(layer_size)
        network.train_network(train_zip, beta, num_epoch, epsilon, batch_size)
        accuracy = network.count_accuracy(valid_set[0], valid_set[1])
        print(f"Network{layer_size}\taccuracy:{accuracy}")
        rated[network] = accuracy
    return max(rated.keys(), key=rated.get)

# hiperparametry
batch_size = 32
epsilon = 0.000001
num_epoch = 30
beta = 0.01
list_layers_size = [[784, 32, 10], [784, 64, 10], [784, 128, 10], [784, 512, 10], [784, 512, 64, 10], [784, 128, 32, 10]]
if __name__ == "__main__":
    X_train, y_train, X_vali, y_vali = get_train_validation_data(0.8)
    X_test, y_test = get_test_data()
    data_zip = [list(zip(X_train, y_train)), [X_vali, y_vali]]
    network = get_best_layer_size(list_layers_size, data_zip, beta, num_epoch, epsilon, batch_size)
    print("-----------------------------------------------------")
    print(f"Best network: size{network.layer_size}\t accuracy")
    accuracy = network.count_accuracy(X_test, y_test)
    print(f"Accuracy after test set: {accuracy}")