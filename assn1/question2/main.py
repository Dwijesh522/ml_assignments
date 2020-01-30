import pandas as pd
import stochastic_gradient_descent as sgd
import plotting
import numpy as np
import math

def sample():
    # number of training examples
    m = 1000000
    theta_not = np.array([[3], [1], [2]])
    # sampling x1 ~ N(3, 4)
    x1 = np.random.normal(3, math.sqrt(4), (m, 1))
    # sampling x2 ~ N(-1, 4)
    x2 = np.random.normal(-1, math.sqrt(4), (m, 1))
    # getting X matrix: each sample in a row, so (m x 3) matrix
    x0 = np.ones((m, 1))
    X = np.concatenate((x0, x1, x2), axis=1)
    # getting mu and sigma to sample y
    mu = np.dot(X, theta_not)
    variance = 2
    # sampling Y ~ N( 3+x1+2x2, 2 )
    Y = np.random.normal(mu, math.sqrt(variance))
    # initializing theta space with zeros
    theta = np.zeros((X.shape[1], 1))
    return X, theta, Y, theta_not


if __name__ == "__main__":

    X, theta, Y, theta_not = sample()

    # shape of x:        1000000 x 3
    # shape of y:        1000000 x 1
    # shape of theta:    3 x 1
    # default learning rate: 0.009

    theta, theta0_space, theta1_space, theta2_space = sgd.stochastic_gradient_descent(X, theta, Y, 0.001, 0.004, 1000000)
    print("optimal theta:\n {}".format(theta))

    plotting.plot_training_data_hypothesis(X, theta, Y)
    plotting.plot_theta_movement(theta0_space, theta1_space, theta2_space)
#    plotting.plot_cost_function(cost_theta_samples[:, 0], cost_theta_samples[:, 1], cost_theta_samples[:, 2])
#    plotting.draw_contours(X, theta, Y, cost_theta_samples[:, 0], cost_theta_samples[:, 1])
