import numpy as np
from scipy.special import expit

import sys
sys.path.append("../../")

def npll_L(params, L, network_adj_mat):
    '''
    Negative pseudo log-likelihood of getting the observed binary L values 
    given the parameters, assuming a logistic regression parametric model. 
    
    Args:
        params: a length 2 numpy array of the form [intercept, coefficient].
        L: a length n_units vector of observed L values.
        network_adj_mat: a n_units x n_units numpy array representing the 
            adjacency matrix of the network.
    
    Returns:
        The negative pseudo log-likelihood of the observed L values given the 
        parameters.
    '''
    pL1 = expit(params[0] + params[1]*(L@network_adj_mat)) # a length n vector.
    pL = L*pL1 + (1-L)*(1-pL1)
    # the expit() function outputs 0.0 when the input is reasonably small, so 
    # we replace 0 with a small const to ensure numerical stability
    pL = np.where(pL == 0, 1e-10, pL) 
    return -np.sum(np.log(pL))

def npll_L_continuous(params, L, network_adj_mat):
    '''
    Negative pseudo log-likelihood of getting the observed continuous L values
    given the parameters, assuming a normal distribution parametric model.
    
    Args:
        params: a length 3 numpy array of the form [intercept, coefficient, sigma].
        L: a length n_units vector of observed L values.
        network_adj_mat: a n_units x n_units numpy array representing the 
            adjacency matrix of the network.
            
    Returns:
        The negative pseudo log-likelihood of the observed L values given the 
        parameters.
    '''
    mu = params[0] + params[1] * (L@network_adj_mat)
    sigma = params[2]
    # negative pseudo log-likelihood of normal distribution
    return np.sum(0.5 * (np.log(2 * np.pi) + np.log(sigma**2) + (L - mu)**2 / sigma**2))

def npll_Y(params, L, A, Y, network_adj_mat):
    '''
    Negative pseudo log-likelihood of getting the observed binary Y values
    given the parameters, assuming a logistic regression parametric model.
    
    Args:
        params: a length 6 numpy array of the form [intercept, L_coefficient, A_coefficient, 
                L_neighbor_coefficient, A_neighbor_coefficient, Y_neighbor_coefficient]
        L: a length n_units vector of observed L values
        A: a length n_units vector of observed A values
        Y: a length n_units vector of observed Y values
        network_adj_mat: a n_units x n_units numpy array representing the 
            adjacency matrix of the network.
    
    Returns:
        The negative pseudo log-likelihood of the observed Y values given the 
        parameters.
    '''
    pY1 = expit((params[0] + params[1]*L + params[2]*A + 
                 params[3]*(L@network_adj_mat) + 
                 params[4]*(A@network_adj_mat) + 
                 params[5]*(Y@network_adj_mat)))
    pY = Y*pY1 + (1-Y)*(1-pY1)
    # the expit() function outputs 0.0 when the input is reasonably small, so 
    # we replace 0 with a small const to ensure numerical stability
    pY = np.where(pY == 0, 1e-10, pY)
    return -np.sum(np.log(pY))

def estimate_causal_effects_U_U(network_adj_mat, A_value, params_L, params_Y, 
                                burn_in=200, n_simulations=100, 
                                gibbs_select_every=3, L_is_continuous=False):
    '''
    Evaluate Y(A=A_value) in a network using Gibbs sampling, assuming the L, 
    A, and Y layers are connected by unidirected edges. The implementation 
    follows the "Gibbs Sampler I" algorithm in the original Auto-G paper.
    
    Y(A=A_value) is the average value of Y_i across all units i in the network 
    when all A_i are intervened to be A_value. 
    
    When true parameters are passed in, this method is used to evaluate the
    true Y(A=A_value) in the network. When estimated parameters are 
    passed in, this method is used to evaluate the estimated Y(A=A_value)
    in the network.
    
    Args:
        network_adj_mat: a n_units x n_units numpy array representing the adjacency matrix of the network.
        A_value: the value of A to intervene on.
        params_L: parameters of the L layer. If L_continuous_data is True,
            params_L is a length 3 numpy array of the form [intercept, coefficient, sigma].
            Otherwise, params_L is a length 2 numpy array of the form [intercept, coefficient].
        params_Y: parameters of the Y layer. A length 6 numpy array.
        burn_in: number of iterations to burn-in the Markov Chain. Default is 200.
        n_simulations: number of rows in matrix_Ys, i.e., how many realizations of Y to sample. Default is 100.
        gibbs_select_every: thin the Markov Chain for every N iteration. Default is 3.
        L_is_continuous: a boolean indicating whether the L layer is continuous. Default is False.
        
    
    Returns:
        For each realization of Y, i.e., each row of matrix_Ys, we calculate the
        average of Y_i across all units i in the network when all A_i are 
        intervened to be A_value. We return the average of these averages.
    '''
    # initialize random L and Y values
    Y = np.random.binomial(1, 0.5, len(network_adj_mat))
    if L_is_continuous:
        L = np.random.normal(size=len(network_adj_mat))
    else:
        L = np.random.binomial(1, 0.5, len(network_adj_mat))

    # set A based on the intervention value A_value
    A = [A_value] * len(L)  
    
    # store the Y values in this matrix after burn-in and thinning
    # matrix_Ys is a list of length K, where each element is a length n_units vector
    matrix_Ys = []

    # run the Gibbs sampler for burn_in +  n_simulations*gibbs_select_every iterations, 
    # where at each iteration we iterate over all units in the network.
    for m in range(burn_in + n_simulations*gibbs_select_every):
        for i in range(len(network_adj_mat)):
            
            # at each iter, we do two things:
            # 1) update L[i] 
            if L_is_continuous:
                # params_L is length three when L_continuous_data is True
                mu = params_L[0] + params_L[1] * np.dot(L, network_adj_mat[i, :])
                sigma = params_L[2]
                L[i] = np.random.normal(mu, sigma)
            else:
                # params_L is length two when L_continuous_data is False
                pLi_given_rest = expit(params_L[0] + params_L[1]*np.dot(L, network_adj_mat[i, :]))
                L[i] = np.random.binomial(1, pLi_given_rest)

            pYi_given_rest = expit(params_Y[0] + params_Y[1]*L[i] + params_Y[2]*A[i] +
                                   params_Y[3]*np.dot(L, network_adj_mat[i, :]) +
                                   params_Y[4]*np.dot(A, network_adj_mat[i, :]) +
                                   params_Y[5]*np.dot(Y, network_adj_mat[i, :]))
            
            # 2) update Y[i]
            # sample a new value for Y[i] based on the conditional probability
            Y[i] = np.random.binomial(1, pYi_given_rest)

        if m > burn_in and m % gibbs_select_every == 0:
            # store the Y values after burn-in and thinning 
            matrix_Ys.append(Y.copy())

    return np.mean(matrix_Ys)
