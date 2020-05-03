import numpy as np

def load_data(file_path):
    """
    Read data from file_path
    :param data_path: link to data file
    :return:
        X: 2d-array features, each row each data-point
        y: 1d-array values, corresponding to data-points
    """
    with open('death_rate_data.txt') as f:
        data = np.loadtxt('death_rate_data.txt')

    X = data[:, 1:-1]
    y = data[:, -1]
    print(type(X))
    return X, y


def normalize_and_add_bias(X):
    """
    Normalize use "Feature Scaling"
    and add bias to features matrix X
    :param X: features matrix X
    :return: normalized and added bias X
    """
    X_max = np.max(X, axis = 0)
    X_min = np.min(X, axis = 0)
    X_normalized = (X - X_min)/(X_max - X_min)
    bias = np.ones((X.shape[0], 1))
    X_done = np.c_[bias, X_normalized]

    return X_done

def train_test_split(X, y):
    """
    Split data (X, y) into train set(50) and test set(10)
    ++ shuffle
    :param X:
    :return: X_train, y_train, X_test, y_test
    """
    X_train, y_train = X[:50, :], y[:50]
    X_test, y_test = X[50:, :], y[50:]

    return X_train, y_train, X_test, y_test