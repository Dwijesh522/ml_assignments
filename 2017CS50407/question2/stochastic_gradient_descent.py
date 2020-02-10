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
def stochastic_gradient_descent(X, theta, Y, learning_rate, converging_epsilon, batch_size): 

    # number of training examples
    m = X.shape[0]
    batches = m//batch_size
    cummulative_band_width = 1
    found_optimal_average_gradient = 0.0012

    # storing thetas as we modify them from plotting
    theta0_space = np.array([0])
    theta1_space = np.array([0])
    theta2_space = np.array([0])


    i = 0
    while(True):

        # shuffle data
#        training_data = np.concatenate((X, Y), axis=1)
#        np.random.shuffle(training_data)
#        X = training_data[:, 0:training_data.shape[1]-1]
#        Y = training_data[:, training_data.shape[1]-1:]

        cummulative_cost=0
        cummulative_gradient=0
        for b in range(batches):
            
            Xb = X[ b*batch_size : (b+1)*batch_size, :]
            Yb = Y[ b*batch_size : (b+1)*batch_size, :]
            
            current_gradient = gradient_cost(Xb, theta, Yb)
            current_cost = cost(Xb, theta, Yb)
            
            theta = theta - learning_rate * current_gradient
            
            # storing current value to theta space
            theta0_space = np.concatenate((theta0_space, theta[0]))
            theta1_space = np.concatenate((theta1_space, theta[1]))
            theta2_space = np.concatenate((theta2_space, theta[2]))

            # storing cost after every 10th batch
            cummulative_cost += current_cost
            cummulative_gradient += math.sqrt(np.dot(current_gradient.T, current_gradient))
            
            if((b)%cummulative_band_width == 0):
                print("{} {} {}".format(theta[0][0], theta[1][0], theta[2][0]))
                print(  "epoch {}, batch: {}, cummulative cost: {} cummulative gradient: {}".format(i, b, 
                        cummulative_cost/cummulative_band_width, cummulative_gradient/cummulative_band_width))
                cummulative_gradient = 0
                cummulative_cost = 0
        print(cummulative_gradient/cummulative_band_width)
        # if average of last few gradients of the signal is with in an average then stop
        if(abs(cummulative_gradient/cummulative_band_width - found_optimal_average_gradient) <= converging_epsilon):
            break
        i += 1
    
    return theta, theta0_space, theta1_space, theta2_space
