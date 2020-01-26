import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

# J(theta) = 1/(2m) * (Y - X.theta)^ * (Y - X.theta)
# returns cost at a given theta
def cost(X, theta, Y):
    temp = Y - np.dot(X, theta)
    return np.dot(temp.T, temp) * (1/Y.shape[0]) * (1/2)

# d(J(theat))/d(theta) = 1/m * ( (X^).X.theta - X^.Y)
# returns gradient of cost functin at given theta
def gradient_cost(X, theta, Y):
    return (1/Y.shape[0]) * ( np.dot( np.dot(X.T, X), theta ) - np.dot(X.T, Y) )

# returns theta that minimizes the cost function
# also returns list(theta, cost)
def batch_gradient_descent(X, theta, Y, learning_rate):
    cost_theta_samples = np.array( [[theta[0][0], theta[1][0], cost(X, theta, Y)[0][0] ]])
    for i in range(20000):
        theta = theta - learning_rate * gradient_cost(X, theta, Y)
        if(i%4000 == 0):    print("iteration {} cost {}".format(i+1, cost(X, theta, Y)))
        if(i%100 == 0):
            cost_theta_samples = np.concatenate( (cost_theta_samples, np.array( [[ theta[0][0], theta[1][0], cost(X, theta, Y)[0][0] ]])))

    return [theta, cost_theta_samples]

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

# drawing training data and optimal hypothesis function
def plot_training_data_hypothesis(X, theta, Y):
    hypothesis = np.dot(X, theta)
    plt.scatter(X[:, 0], Y[:, 0], label = 'training data')
    plt.plot(X[:, 0], hypothesis[:, 0], label = 'hypothesis function')
    plt.xlabel('acidity')
    plt.ylabel('desity')
    plt.legend()
    plt.savefig('data.png', dpi = 1000, bbox_inches = 'tight')

# draw cost functin for iterated thetas
def plot_cost_function(theta0, theta1, cost_value):
    fig = plt.figure()
    # 1*1 grid with 1 subplot
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=340)
    ax.set_xlabel('theta_0')
    ax.set_ylabel('theta_1')
    ax.set_zlabel('cost')
    # add point one by one
    for i in range(theta0.shape[0]):
        ax.scatter(theta0[i], theta1[i], cost_value[i])
        plt.pause(0.2)
    fig.savefig('cost.png', dpi = 1000, bbox_inches = 'tight')
    plt.show()

if __name__ == "__main__":

    X, theta, Y = init()

    # shape of x:        100 x 2
    # shape of y:        100 x 1
    # shape of theta:    2 x 1

    theta, cost_theta_samples = batch_gradient_descent(X, theta, Y, 0.009)
    print(theta)

    plot_training_data_hypothesis(X, theta, Y)
    plot_cost_function(cost_theta_samples[:, 0], cost_theta_samples[:, 1], cost_theta_samples[:, 2])
