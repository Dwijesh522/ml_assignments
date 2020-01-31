import numpy as np
import pandas as pd
import math
import newtons_method as nm

# use   np.multply for element wise multiplication
#       np.sum for summing along direction

def init():
    # reading x, y
    x_path = "/home/dwijesh/Documents/sem6/ml/assns/datasets/ass1_data/q3/logisticX.csv"
    y_path = "/home/dwijesh/Documents/sem6/ml/assns/datasets/ass1_data/q3/logisticY.csv"
    X = pd.read_csv(x_path, ',', header=None, usecols=[0, 1]).to_numpy()
    Y = pd.read_csv(y_path, ',', header=None, usecols=[0]).to_numpy()
    # appending ones to the features to incorporte intercept term
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)
    # initializing theta space with zeros
    theta = np.zeros((X.shape[1], 1))
    return [X, theta, Y]

if __name__ == '__main__':
    
    X, theta, Y = init()

    # X         : 100 x 3
    # theta     : 3 x 1
    # Y         : 100 x 1
    
    theta = nm.newtons_method(X, theta, Y)
