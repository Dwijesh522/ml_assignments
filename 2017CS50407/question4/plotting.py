import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

# drawing training data and optimal hypothesis function
def plot_training_data_hypothesis(X, Y):
    ax = plt.subplot(111)
    for i in range(Y.shape[0]):
        if(Y[i][0] == 'Alaska'):
            ax.scatter(X[i][0], X[i][1], color='red')
        else:
            ax.scatter(X[i][0], X[i][1], color='blue')

    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
#    plt.savefig('data.png', dpi = 1000, bbox_inches = 'tight')
    print("training data has been plotted...")
    return ax

def plot_decision_boundry(phy, mu_0, mu_1, sigma_0, sigma_1, ax):
    
    # inverse of sigma matrices
    sigma_0_inv = np.linalg.inv(sigma_0)
    sigma_1_inv = np.linalg.inv(sigma_1)
    # determinant of sigma matrices
    sigma_0_det = np.linalg.det(sigma_0)
    sigma_1_det = np.linalg.det(sigma_1)
    
    x1, x2 = np.meshgrid(np.arange(-2, 2, 0.001), np.arange(-2, 2, 0.001))
    
    # finding coefficient of eac power of x for decision boundry equation
    # 1st: constant term belongs to real number
    constant_term = np.dot( np.dot(mu_1.T, sigma_1_inv), mu_1)[0][0] - np.dot( np.dot( mu_0.T, sigma_0_inv), mu_0)[0][0] + math.log( ((1-phy)/phy)**2 ) + math.log( sigma_1_det/sigma_0_det)
    # 2nd: linear coefficient (1 x 2)
    linear_coefficient = 2 * (np.dot(mu_0.T, sigma_0_inv) - np.dot(mu_1.T, sigma_1_inv))
    # 3rd: quadratic coefficient (2 x 2)
    quadratic_coefficient = sigma_1_inv - sigma_0_inv
    # some temporary paramters
    q11 = quadratic_coefficient[0][0]
    q12 = quadratic_coefficient[0][1]
    q21 = quadratic_coefficient[1][0]
    q22 = quadratic_coefficient[1][1]
    l1 = linear_coefficient[0][0]
    l2 = linear_coefficient[0][1]
    # writting the complete equation of decision boundry:
    z =     q11*(x1**2) + q22*(x2**2) + (q21 + q12)*x1*x2 + 2*l1*x1 + 2*l2*x2 + constant_term
    
    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            if(abs(z[i][j]) <= 0.001):
                ax.scatter(x1[i][j], x2[i][j], color='black', s=1)
    
#    plt.savefig(name, dpi=1000, bbox_inches='tight')
    print("hypothesis plotted")
    return ax
