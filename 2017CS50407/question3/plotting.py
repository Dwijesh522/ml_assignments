import matplotlib.pyplot as plt
import numpy as np

# drawing training data and optimal hypothesis function
def plot_training_data_hypothesis(X, theta, Y):
    ax = plt.subplot(111)
    for i in range(Y.shape[0]):
        if(Y[i][0] == 0):

            ax.scatter(X[i][1], X[i][2], color='red')
        else:
            ax.scatter(X[i][1], X[i][2], color='blue')

    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
#    plt.savefig('data.png', dpi = 1000, bbox_inches = 'tight')
    print("training data has been plotted...")
    return ax

def plot_hypothesis(X, theta, ax):
    # equation of plane: 00 + 01x1 + 02x2 = 0
#    ax = fig.add_subplot(2, 2, i+2, projection='3d')
    x1 = range(-2, 4)
    x2 = ((theta[0][0] + theta[1][0]*x1)*(-1)) / theta[2][0]
    ax.plot(x1, x2, color='green')
    
#    ax.view_init(elev=10, azim=42)
    plt.savefig("data.png", dpi=1000, bbox_inches='tight')
    print("hypothesis plotted")
