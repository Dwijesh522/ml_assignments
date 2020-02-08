import numpy as np
import math
import plotting
import copy

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
        hessian_matrix += np.dot( np.reshape(X[i], (-1, 1)), np.reshape(inter1[i], (1, -1)) )

    return -1*hessian_matrix

# returns gradient of ll(0) at theta
def gradient(X, theta, Y):
    return  np.reshape( np.sum( (np.multiply( (Y - sig(np.dot(X, theta))) , X)) , axis=0), (-1, 1) )

# returns sum: yi log(sig(0Txi)) + (1-yi) log(1- sig(0Txi))
def log_likelihood(X, theta, Y):
    h0x = sig(np.dot(X, theta))
    first_operand = np.multiply(Y, np.log(h0x))
    second_operand = np.multiply((1-Y), np.log(1-h0x))
    return np.sum(first_operand+second_operand, axis=0)

# returns optimal theta learnt for 
# logistic regression using newton's method
def newtons_method(X, theta, Y, fig, ax):
    old_log_likelihood = log_likelihood(X, theta, Y)
    iter_count=0
    while(True):
        # 0 = 0 - (inv(H)) * d(l(0))/d(0)
        theta = theta - np.dot( np.linalg.inv(hessian(X, theta)), gradient(X, theta, Y) )
        new_log_likelihood = log_likelihood(X, theta, Y)
        if( abs(new_log_likelihood - old_log_likelihood) <= 1e-10):   break
        else:   old_log_likelihood = new_log_likelihood
        iter_count += 1
        if(iter_count == 100):  break
    print("number of iteractions in newton's method: {}".format(iter_count))
    plotting.plot_hypothesis(X, theta, fig, ax)
    print("theta:\n{}".format(theta))
    return theta
