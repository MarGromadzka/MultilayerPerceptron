import numpy as np
from mlxtend.data import loadlocal_mnist
import platform

def get_train_validation_data(ratio):
    """ zwraca treningowe znormalizowane dane, X to obrazy, y to odpowiadająca mu cyfry
    ratio - jaka część będzie treningowa (z 60000)"""
    if not platform.system() == 'Windows':
        X, y = loadlocal_mnist(
                images_path='train-images-idx3-ubyte', 
                labels_path='train-labels-idx1-ubyte')
    else:
        X, y = loadlocal_mnist(
                images_path='train-images.idx3-ubyte', 
            labels_path='train-labels.idx1-ubyte')

    X = X.astype(float) / 255
    X_train = X[0:int(ratio*len(X))]
    y_train = y[0:int(ratio*len(y))]
    X_validation = X[int(ratio*len(X)):len(X)]
    y_validation = y[int(ratio*len(y)):len(X)]

    return X_train, y_train, X_validation, y_validation

def get_test_data():
    """ zwraca testowe znormalizowane dane, X to obrazy, y to odpowiadająca mu cyfry"""
    if not platform.system() == 'Windows':
        X, y = loadlocal_mnist(
                images_path='t10k-images-idx3-ubyte', 
                labels_path='t10k-labels-idx1-ubyte')
    else:
        X, y = loadlocal_mnist(
                images_path='t10k-images.idx3-ubyte', 
            labels_path='t10k-labels.idx1-ubyte')

    X = X.astype(float) / 255

    return X,y
