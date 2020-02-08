import numpy as np
import pandas as pd
import plotting
import gla
import matplotlib.pyplot as plt

# use   np.multply for element wise multiplication
#       np.sum for summing along direction

# +-----------------------------------+
# |convention:   0 -----> Alaska, red |
# |              1 -----> Canada, blue|
# +-----------------------------------+

def init():
    # reading x, y
    x_path = "/home/dwijesh/Documents/sem6/ml/assns/datasets/ass1_data/q4/q4x.dat"
    y_path = "/home/dwijesh/Documents/sem6/ml/assns/datasets/ass1_data/q4/q4y.dat"
    X = pd.read_csv(x_path, '  ', header=None, usecols=[0, 1]).to_numpy()
    Y = pd.read_csv(y_path, ',', header=None, usecols=[0]).to_numpy()
    # separating the training points according to two classes of Y: 0, 1
    indices_0, _ = np.where(Y == 'Alaska')
    indices_1, _ = np.where(Y == 'Canada')
    X_0 = X[indices_0, :]
    X_1 = X[indices_1, :]
    Y_0 = np.zeros((indices_0.shape[0], 1))
    Y_1 = np.ones((indices_1.shape[0], 1))
    # initializing theta space with zeros
    return [X_0, X_1, Y_0, Y_1, X, Y]

if __name__ == '__main__':
    
    X_0, X_1, Y_0, Y_1, X, Y = init()

    # X         : 100 x 2
    # Y         : 100 x 1
    
    fig, ax = plotting.plot_training_data_hypothesis(X, Y)
    phy, mu_0, mu_1, sigma_0, sigma_1 = gla.gaussian_discriminant_analysis(X_0, X_1, Y_0, Y_1)
    sigma = ((sigma_0 * Y_0.shape[0]) + (sigma_1 * Y_1.shape[0]))/(Y_0.shape[0]+Y_1.shape[0])

    print("phy:\n{}".format(phy))
    print("mu_0\n{}".format(mu_0))
    print("mu_1\n{}".format(mu_1))
    print("sigma_0\n{}".format(sigma_0))
    print("sigma_1\n{}".format(sigma_1))
    print("sigma:\n{}".format(sigma))

    # case 1 when both the sigmas are same
    fig, ax = plotting.plot_decision_boundry(phy, mu_0, mu_1, sigma, sigma, fig, ax)
    # case2: when both the sigmas are different
    fig, ax = plotting.plot_decision_boundry(phy, mu_0, mu_1, sigma_0, sigma_1, fig, ax)
    plt.savefig("data.png", dpi=1000, bbox_inches='tight')
