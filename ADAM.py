import numpy as np

pset = [[[5.7, 4.4, 1.5, 0.4], 1], [[5.1, 3.8, 1.6, 0.2], 1], [[5.0, 3.4, 1.5, 0.2], 1], [[5.4, 3.9, 1.7, 0.4], 1],
        [[4.9, 3.1, 1.5, 0.1], 1], [[4.9, 3.1, 1.5, 0.1], 1], [[4.4, 2.9, 1.4, 0.2], 1], [[5.4, 3.7, 1.5, 0.2], 1],
        [[4.6, 3.6, 1.0, 0.2], 1], [[5.0, 3.5, 1.6, 0.6], 1], [[5.1, 3.3, 1.7, 0.5], 1], [[5.2, 3.4, 1.4, 0.2], 1],
        [[5.2, 4.1, 1.5, 0.1], 1], [[4.4, 3.0, 1.3, 0.2], 1], [[5.8, 4.0, 1.2, 0.2], 1], [[5.5, 3.5, 1.3, 0.2], 1],
        [[5.1, 3.5, 1.4, 0.2], 1], [[4.5, 2.3, 1.3, 0.3], 1], [[4.7, 3.2, 1.3, 0.2], 1], [[4.8, 3.0, 1.4, 0.3], 1],
        [[5.1, 3.8, 1.5, 0.3], 1], [[5.0, 3.2, 1.2, 0.2], 1], [[4.8, 3.1, 1.6, 0.2], 1], [[5.0, 3.3, 1.4, 0.2], 1],
        [[4.9, 3.0, 1.4, 0.2], 1], [[4.8, 3.4, 1.6, 0.2], 1], [[5.0, 3.6, 1.4, 0.2], 1], [[4.3, 3.0, 1.1, 0.1], 1],
        [[4.4, 3.2, 1.3, 0.2], 1], [[4.6, 3.4, 1.4, 0.3], 1], [[5.0, 3.0, 1.6, 0.2], 1], [[4.6, 3.1, 1.5, 0.2], 1],
        [[5.8, 2.6, 4.0, 1.2], -1], [[5.5, 2.5, 4.0, 1.3], -1], [[6.3, 3.3, 4.7, 1.6], -1], [[6.7, 3.1, 4.7, 1.5], -1],
        [[6.4, 2.9, 4.3, 1.3], -1], [[5.8, 2.7, 4.1, 1.0], -1], [[6.6, 2.9, 4.6, 1.3], -1], [[5.7, 2.9, 4.2, 1.3], -1],
        [[6.1, 3.0, 4.6, 1.4], -1], [[6.2, 2.2, 4.5, 1.5], -1], [[6.3, 2.5, 4.9, 1.5], -1], [[6.1, 2.8, 4.7, 1.2], -1],
        [[6.1, 2.8, 4.0, 1.3], -1], [[5.6, 3.0, 4.1, 1.3], -1], [[5.9, 3.2, 4.8, 1.8], -1], [[5.0, 2.3, 3.3, 1.0], -1],
        [[6.2, 2.9, 4.3, 1.3], -1], [[6.9, 3.1, 4.9, 1.5], -1], [[5.7, 2.8, 4.1, 1.3], -1], [[6.3, 2.3, 4.4, 1.3], -1],
        [[5.9, 3.0, 4.2, 1.5], -1], [[6.0, 2.7, 5.1, 1.6], -1], [[6.5, 2.8, 4.6, 1.5], -1], [[5.6, 2.9, 3.6, 1.3], -1],
        [[5.2, 2.7, 3.9, 1.4], -1], [[5.6, 3.0, 4.5, 1.5], -1]]


def f(x, attri, b):
    return np.dot(np.transpose(attri), x) - b


def calc_gradient(attributes, point):
    gradient = []
    for i in range(len(attributes) - 1):  # pochodna cząstkowa po w
        partial_derivative = 2 * 0.1 * attributes[i]
        if 1 - point[1] * f(point[0], attributes[:-1], attributes[-1]) > 0:
            partial_derivative -= point[1] * point[0][i]
        gradient.append(partial_derivative)
    b_partial_derivative = 0  # pochodna cząstkowa po b
    if 1 - point[1] * f(point[0], attributes[:-1], attributes[-1]) > 0:
        b_partial_derivative += point[1]
    gradient.append(b_partial_derivative)
    return np.array(gradient)


def get_random_vec(data, weight, batch_size, gamma):
    indexes = np.random.randint(len(data), size=batch_size)
    points = [data[indexes[i]] for i in range(batch_size)]
    avg_grad = [0 for i in range(len(weight))]
    for point in points:
        avg_grad = np.add(avg_grad, calc_gradient(weight, point))
    return np.divide(avg_grad, batch_size)


def calc_avg(gradient_list):
    if len(gradient_list) != 0:
        avg = np.mean(np.square(gradient_list), axis=0)
        avg = np.where(avg == 0, 1, avg)
        return np.sqrt(avg)
    else:
        return 1


def adam_algorithm(beta, weight, data, max_iter, epsilon, batch_size, gamma):
    grad_history = []
    weight_before = weight
    for i in range(max_iter):
        random_vec = get_random_vec(data, weight, batch_size, gamma)
        bg = beta * np.divide(random_vec, calc_avg(grad_history))
        grad_history.append(random_vec)
        momentum = np.multiply(np.subtract(weight, weight_before), gamma)
        weight_before = weight
        weight = np.add(np.subtract(weight, bg), momentum)
        if abs(beta * np.linalg.norm(random_vec)) < epsilon:
            return weight
    return weight


random_weight = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
gamma = 0.9
batch_size = 4
for i in range(0, 100):
    sgd = adam_algorithm(0.1, random_weight, pset, 1000, 0.00001, batch_size, gamma)
    print(sgd)
