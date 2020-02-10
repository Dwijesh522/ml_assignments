import numpy as np
import pandas as pd
import math
import newtons_method as nm
import plotting

# use   np.multply for element wise multiplication
#       np.sum for summing along direction

# returns expactation list corresponding to each feature in dataset
def expectation(X):
    return np.sum(X, axis=0)/X.shape[0]

# returns standard deviation corresponding to each feature in dataset
def standard_deviation(X, mu):
    sigma_2 = expectation(X**2) - mu**2
    return np.array([math.sqrt(obj) for obj in sigma_2 ])
# returns normalized features, mu, sigma for each features
def normalize(X):
    mu = expectation(X)
    sigma = standard_deviation(X, mu)
    X = (X-mu)/sigma
    return X, mu, sigma

def init():
    # reading x, y
    x_path = "/home/dwijesh/Documents/sem6/ml/assns/datasets/ass1_data/q3/logisticX.csv"
    y_path = "/home/dwijesh/Documents/sem6/ml/assns/datasets/ass1_data/q3/logisticY.csv"
    X = pd.read_csv(x_path, ',', header=None, usecols=[0, 1]).to_numpy()
    Y = pd.read_csv(y_path, ',', header=None, usecols=[0]).to_numpy()
    # normalize x features
    X, mu, sigma = normalize(X)
    # appending ones to the features to incorporte intercept term
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)
    # initializing theta space with zeros
    theta = np.zeros((X.shape[1], 1))
    return [X, theta, Y]

# prints the hypothesis values and correct values
def print_predictions(X, theta, Y):
    predictions = nm.sig(np.dot(X, theta))
    correct, wrong = [0, 0]
    for i in  range(Y.shape[0]):
        if(predictions[i] >= 0.5):
            if(Y[i]==1):    correct += 1
            else:           wrong += 1
        else:
            if(Y[i]==0):    correct += 1
            else:           wrong += 1
    print("correctly classified: {}".format(correct))
    print("wrongly classified: {}".format(wrong))

if __name__ == '__main__':
    
    X, theta, Y = init()

    # X         : 100 x 3
    # theta     : 3 x 1
    # Y         : 100 x 1
    
    ax = plotting.plot_training_data_hypothesis(X, theta, Y)
    theta = nm.newtons_method(X, theta, Y, ax)
    print_predictions(X, theta, Y)
