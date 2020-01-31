import numpy as np
import math

# returns sigmoid over a one dimensional array
def sig(x):
    return np.reshape( np.array( [1/(1+math.exp((-1)*obj)) for obj in x] ), (-1, 1))

# returns derivative of sigmoid function at one dimensional array elements
def sig_derivative(x):
    return np.reshape( np.array( [math.exp((-1)*obj)/((1+math.exp((-1)*obj))**2 ) for obj in x] ), (-1, 1) )

# returns hessian matrix of ll(0)
def hessian(X, theta):
    m = X.shape[0]          # of training examples
    n = theta.shape[0]      # of features in x
    inter1 = np.multiply( sig_derivative(np.dot(X, theta)), X)
    hessian_matrix = np.zeros((n, n))
    # performing matrix summation
    for i in range(m):
        hessian_matrix += np.dot( np.reshape(X[i], (n, 1)), np.reshape(inter1[i], (1, -1)) )

    return hessian_matrix

# returns gradient of ll(0) at theta
def gradient(X, theta, Y):
    return  np.reshape( np.sum( (np.multiply( (Y - sig(np.dot(X, theta))) , X)) , axis=0), (-1, 1) )

# returns optimal theta learnt for 
# logistic regression using newton's method
def newtons_method(X, theta, Y):
    
    for i in range(10):
        # 0 = 0 - (inv(H)) * d(l(0))/d(0)
        print(gradient(X, theta, Y))
        theta = theta - np.dot( np.linalg.inv(hessian(X, theta)), gradient(X, theta, Y) )

    print(theta)
    return theta
