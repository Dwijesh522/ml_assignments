import numpy as np
# returns phy, mu_0, mu_1, sigma_0, simgma_1:
# The Gaussian Discirminant Analysis parameters
def gaussian_discriminant_analysis(X_0, X_1, Y_0, Y_1):
    # number of instances of each class
    indicator_0, indicator_1 = Y_0.shape[0], Y_1.shape[0]
    # phy parameter
    phy = indicator_1/(indicator_0 + indicator_1)
    # mu_0 and mu_1
    mu_0 = np.reshape( ( np.sum(X_0, axis=0) / indicator_0 ), (-1, 1))
    mu_1 = np.reshape( ( np.sum(X_1, axis=0) / indicator_1 ), (-1, 1))
    # sigma_0 and sigma_1 of size n x n where n is number of features: 2
    sigma_0 = np.zeros((X_0.shape[1], X_0.shape[1]))
    sigma_1 = np.zeros((X_1.shape[1], X_1.shape[1]))
    # sigma_0
    # [ ... (x(i)-mu_0) ...].T
    V_0 = (X_0.T - mu_0).T
    for i in range(V_0.shape[0]):
        sigma_0 += np.dot( np.reshape(V_0[i], (-1, 1)), np.reshape( V_0[i], (1, -1)) )
    sigma_0 /= indicator_0
    # sigma_1
    # [ ... (x(i)-mu_1) ...].T
    V_1 = (X_1.T - mu_1).T
    for i in range(V_1.shape[0]):
        sigma_1 += np.dot( np.reshape(V_1[i], (-1, 1)), np.reshape( V_1[i], (1, -1)) )
    sigma_1 /= indicator_1

    return [phy, mu_0, mu_1, sigma_0, sigma_1]
