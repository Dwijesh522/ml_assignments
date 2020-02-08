import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# drawing training data and optimal hypothesis function
def plot_training_data_hypothesis(X, theta, Y):
    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')
    for i in range(Y.shape[0]):
        if(Y[i][0] == 0):
            ax.scatter(X[i][1], X[i][2], Y[i][0], color='red')
        else:
            ax.scatter(X[i][1], X[i][2], Y[i][0], color='blue')

    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_zlabel("y")
#    plt.savefig('data.png', dpi = 1000, bbox_inches = 'tight')
    print("training data has been plotted...")
    return [fig, ax]

def plot_hypothesis(X, theta, fig, ax):
    # equation of plane: 00 + 01x1 + 02x2 = 0
#    ax = fig.add_subplot(2, 2, i+2, projection='3d')
    x1, x2 = np.meshgrid(range(2, 6), range(2, 6))
    z = theta[0][0] + theta[1][0]*x1 + theta[2][0]*x2
    ax.set_zlim(-3, 5)
    ax.plot_surface(x1, x2, z, alpha=0.2)
    
    ax.view_init(elev=10, azim=42)
    plt.savefig("data.png", dpi=1000, bbox_inches='tight')
    print("hypothesis plotted")
