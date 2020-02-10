import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import batch_gradient_descent as bgd

# drawing training data and optimal hypothesis function
def plot_training_data_hypothesis(X, theta, Y):
    hypothesis = np.dot(X, theta)
    plt.scatter(X[:, 1], Y[:, 0], label = 'training data')
    plt.plot(X[:, 1], hypothesis[:, 0], label = 'hypothesis function')
    plt.xlabel('acidity')
    plt.ylabel('desity')
    plt.legend()
    plt.savefig('data.png', dpi = 1000, bbox_inches = 'tight')
    print("training data and hypothesis has been plotted...")

# draw cost function for iterated thetas
#def plot_cost_function(theta0, theta1, cost_value):
#    fig = plt.figure()
#    # 1*1 grid with 1 subplot
#    ax = fig.add_subplot(111, projection='3d')
#    ax.view_init(elev=340)
#    ax.set_xlim(0, 1)
#    ax.set_ylim(0, 0.0015)
#    ax.set_zlim(0, 0.5)
#    ax.set_xlabel('theta_0')
#    ax.set_ylabel('theta_1')
#    ax.set_zlabel('cost')
#    # add point one by one
#    for i in range(theta0.shape[0]):
#        ax.scatter(theta0[i], theta1[i], cost_value[i])
#        plt.pause(0.2)
#    fig.savefig('cost.png', dpi = 1000, bbox_inches = 'tight')
#    print("cost function plotted...")
    plt.show()


# draw contours for each iteration of the gradient descient
def draw_contours(X, theta, Y, theta0_col, theta1_col):
    theta0_mesh, theta1_mesh = np.meshgrid(theta0_col, theta1_col)
    cost_mesh = np.zeros((1, theta0_mesh.shape[1]))

    
    # calculating cost function at each grid position
    for i in range(theta0_mesh.shape[0]):
        temp_cost_list = np.array([[]])
        for j in range(theta0_mesh.shape[1]):
            temp_cost_list = np.concatenate( (temp_cost_list, bgd.cost(X, np.array([ [theta0_mesh[i][j]], [theta1_mesh[i][j]] ] ), Y) ), axis=1)
        cost_mesh = np.concatenate((cost_mesh, temp_cost_list), axis=0)
    cost_mesh = cost_mesh[1:, :]

    x0, x1 = np.meshgrid(np.arange(-2, 3, 0.2), np.arange(-2, 3, 0.2))
    z = np.zeros((1, x0.shape[1]))
    # calculating cost function at each grid position
    for i in range(x0.shape[0]):
        temp_cost_list = np.array([[]])
        for j in range(x0.shape[1]):
            temp_cost_list = np.concatenate( (temp_cost_list, bgd.cost(X, np.array([ [x0[i][j]], [x1[i][j]] ] ), Y) ), axis=1)
        z = np.concatenate((z, temp_cost_list), axis=0)
    z = z[1:, :]
    
    # plotting the contour
    fig, ax = plt.subplots()
    CS = ax.contour(x0, x1, z)
    ax.set_xlabel('theta0')
    ax.set_ylabel('theta1')
    # plotting theta points traversed while applying gradient descent
    for i in range(theta0_mesh.shape[0]):
        ax.set_title("Contour of cost function in theta space, current cost: %1.10f" %cost_mesh[i][i])
        plt.plot(theta0_mesh[i][i], theta1_mesh[i][i], 'ro')
        plt.pause(0.2)

    ax.clabel(CS, inline=1, fontsize=10)
    fig.savefig('contour.png')
    print("contours plotted...")
    plt.show()

# draw contours for each iteration of the gradient descient
def plot_cost_function(X, theta, Y, theta0_col, theta1_col):
    theta0_mesh, theta1_mesh = np.meshgrid(theta0_col, theta1_col)
    cost_mesh = np.zeros((1, theta0_mesh.shape[1]))

    
    # calculating cost function at each grid position
    for i in range(theta0_mesh.shape[0]):
        temp_cost_list = np.array([[]])
        for j in range(theta0_mesh.shape[1]):
            temp_cost_list = np.concatenate( (temp_cost_list, bgd.cost(X, np.array([ [theta0_mesh[i][j]], [theta1_mesh[i][j]] ] ), Y) ), axis=1)
        cost_mesh = np.concatenate((cost_mesh, temp_cost_list), axis=0)
    cost_mesh = cost_mesh[1:, :]

    x0, x1 = np.meshgrid(np.arange(-2, 3, 0.2), np.arange(-2, 3, 0.2))
    z = np.zeros((1, x0.shape[1]))
    # calculating cost function at each grid position
    for i in range(x0.shape[0]):
        temp_cost_list = np.array([[]])
        for j in range(x0.shape[1]):
            temp_cost_list = np.concatenate( (temp_cost_list, bgd.cost(X, np.array([ [x0[i][j]], [x1[i][j]] ] ), Y) ), axis=1)
        z = np.concatenate((z, temp_cost_list), axis=0)
    z = z[1:, :]
    
    # plotting the contour
    fig = plt.figure()
    # 1*1 grid with 1 subplot
    ax = fig.add_subplot(111, projection='3d')
    CS = ax.plot_surface(x0, x1, z, alpha=0.2)
    ax.set_xlabel('theta0')
    ax.set_ylabel('theta1')
    # plotting theta points traversed while applying gradient descent
    for i in range(theta0_mesh.shape[0]):
#        ax.set_title("Contour of cost function in theta space, current cost: %1.10f" %cost_mesh[i][i])
        ax.scatter(theta0_mesh[i][i], theta1_mesh[i][i], cost_mesh[i][i], color='black')
        plt.pause(0.2)

    ax.clabel(CS, inline=1, fontsize=10)
    fig.savefig('cost.png')
    print("contours plotted...")
    plt.show()
