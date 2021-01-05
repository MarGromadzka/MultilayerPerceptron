import numpy as np
from mlxtend.data import loadlocal_mnist
from Layers import Layer
from data_preparation import get_test_data, get_train_validation_data
from Network import Network

X_train, y_train, X_vali, y_vali = get_train_validation_data(0.8)

network = Network([Layer(784, 784, False), Layer(784, 10, True)])
x = network.predict(X_train[0])
print(x)


