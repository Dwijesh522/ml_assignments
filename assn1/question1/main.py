import pandas as pd
import batch_gradient_descent as bgd
import plotting
import numpy as np

def init():
    # reading x, y
    x_path = "/home/dwijesh/Documents/sem6/ml/assns/datasets/ass1_data/q1/linearX.csv"
    y_path = "/home/dwijesh/Documents/sem6/ml/assns/datasets/ass1_data/q1/linearY.csv"
    X = pd.read_csv(x_path, ',', header=None, usecols=[0]).to_numpy()
    Y = pd.read_csv(y_path, ',', header=None, usecols=[0]).to_numpy()
    # appending ones to the features to incorporte intercept term
    ones = np.ones((len(X), 1))
    X = np.concatenate((X, ones), axis=1)
    # initializing theta space with zeros
    theta = np.zeros((X.shape[1], 1))
    return [X, theta, Y]



if __name__ == "__main__":

    X, theta, Y = init()

    # shape of x:        100 x 2
    # shape of y:        100 x 1
    # shape of theta:    2 x 1
    # default learning rate: 0.009

    theta, cost_theta_samples = bgd.batch_gradient_descent(X, theta, Y, 0.009, 0.00005)
    print("optimal theta: {}".format(theta))

    plotting.plot_training_data_hypothesis(X, theta, Y)
    plotting.plot_cost_function(cost_theta_samples[:, 0], cost_theta_samples[:, 1], cost_theta_samples[:, 2])
    plotting.draw_contours(X, theta, Y, cost_theta_samples[:, 0], cost_theta_samples[:, 1])
