import numpy as np
from mlxtend.data import loadlocal_mnist
from Layers import Layer
from data_preparation import get_test_data, get_train_validation_data
from Network import Network

X_train, y_train, X_vali, y_vali = get_train_validation_data(0.8)
X_test, y_test = get_test_data()


network = Network([784, 30, 10])
# network.backward_propagation(X_train[0], y_train[0])
# x = network.predict(X_train[0])
# print(x)
network.train_network(X_train, y_train, 0.01, 30, 0.000001, 32)
network.count_accuracy(X_test, y_test)
