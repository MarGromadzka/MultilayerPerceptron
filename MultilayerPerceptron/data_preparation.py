import numpy as np
from mlxtend.data import loadlocal_mnist


def get_train_data():
    """ zwraca treningowe znormalizowane dane, X to obrazy, y to odpowiadająca mu cyfry"""
    if not platform.system() == 'Windows':
        X, y = loadlocal_mnist(
                images_path='train-images-idx3-ubyte', 
                labels_path='train-labels-idx1-ubyte')
    else:
        X, y = loadlocal_mnist(
                images_path='train-images.idx3-ubyte', 
            labels_path='train-labels.idx1-ubyte')

    X = X.astype(float) / 255
    return X, y

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
