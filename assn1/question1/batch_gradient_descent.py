import numpy as np
import math

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
def batch_gradient_descent(X, theta, Y, learning_rate, converging_threshold): 
    cost_theta_samples = np.array( [[theta[0][0], theta[1][0], cost(X, theta, Y)[0][0] ]])

    i=0
    while(True):
        current_gradient = gradient_cost(X, theta, Y)
        theta = theta - learning_rate * current_gradient
#        print("cost: {}".format(cost(X, theta, Y)))
#        if i == 40: return
        if( math.sqrt(np.dot(current_gradient.T, current_gradient)[0][0]) <= converging_threshold ):   break
        # storing every 100 th cost and theta pair to visualize
        if(i%500 == 0):
#        if i <= 40:
            current_cost = cost(X, theta, Y)
            print("iteration {}, cost: {}".format(i+1, current_cost))
            cost_theta_samples = np.concatenate( (cost_theta_samples, np.array( [[ theta[0][0], theta[1][0], current_cost ]])))
#        else: break
        i += 1
    return [theta, cost_theta_samples]

