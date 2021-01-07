import numpy as np
from mlxtend.data import loadlocal_mnist
from Layers import Layer
from data_preparation import get_test_data, get_train_validation_data
from Network import Network

X_train, y_train, X_vali, y_vali = get_train_validation_data(0.8)

network = Network([Layer(784, 784, True), Layer(784, 20, True), Layer(20, 10, False)])
network.backward_propagation(X_train[0], y_train[0])
# x = network.predict(X_train[0])
# print(x)


