import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# drawing training data and optimal hypothesis function
def plot_training_data_hypothesis(X, theta, Y):
    hypothesis = np.dot(X, theta)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(1000):
        ax.scatter(X[i][1], X[i][2], Y[i][0])

    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_zlabel("y")
    plt.savefig('data.png', dpi = 1000, bbox_inches = 'tight')
    print("training data and hypothesis has been plotted...")

def plot_theta_movement(theta0_space, theta1_space, theta2_space):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(theta0_space, theta1_space, theta2_space)
    ax.set_xlabel("theta0")
    ax.set_ylabel("theta1")
    ax.set_zlabel("theta2")
    plt.savefig("thate_space.png", dpi = 1000, bbox_inches = 'tight')
    print("plotting thata space done...")
