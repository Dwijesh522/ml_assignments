import pandas as pd
import numpy as np
import stochastic_gradient_descent as sgd

if __name__ == '__main__':
    
    theta = np.array([[3], [1], [2]])

    theta_1 = np.array([[2.98841157], [0.98489459], [1.95629993]])

    theta_100 = np.array([[2.99857972], [1.00004456], [1.99967958]])

    theta_10000 = np.array([[2.91204599], [1.01900053], [1.99320696]])

    theta_1000000 = np.array([[3.000254788291396], [1.0002631058712823], [1.999562934111643]])

    path = "/home/dwijesh/Documents/sem6/ml/assns/datasets/ass1_data/q2/q2test.csv"
    training_data = pd.read_csv(path, ',', usecols=[0, 1, 2]).to_numpy()

    # preparing X and Y matrix
    X = training_data[:, 0:2]
    Y = training_data[:, 2:3]
    ones = np.ones((X.shape[0], 1))
    X = np.concatenate((ones, X), axis=1)

    error = sgd.cost(X, theta, Y)
    print("error: {}".format(error))
    
    error = sgd.cost(X, theta_1, Y)
    print("error_1: {}".format(error))
    
    error = sgd.cost(X, theta_100, Y)
    print("error_100: {}".format(error))
    
    error = sgd.cost(X, theta_10000, Y)
    print("error_10000: {}".format(error))
    
    error = sgd.cost(X, theta_1000000, Y)
    print("error_1000000: {}".format(error))
