from scipy.optimize import minimize
from scipy.special import expit
import pandas as pd
import numpy as np
from util import random_network_adjacency_matrix
from concurrent.futures import ProcessPoolExecutor
import networkx as nx

def ring_adjacency_matrix(num_units):

    network = np.zeros((num_units, num_units))
    for i in range(num_units-1):
        network[i, i+1] = 1
        network[i+1, i] = 1

    network[0, num_units-1] = 1
    network[num_units-1, 0] = 1

    return network

def gibbs_sample_L(network, params, burn_in=200):

    # initialize a vector of Ls
    L = np.random.binomial(1, 0.5, len(network))

    # keep sampling an L vector till burn in is done
    for m in range(burn_in):
        for i in range(len(network)):
            pLi_given_rest = expit(params[0] + params[1]*np.dot(L, network[i, :]))
            L[i] = np.random.binomial(1, pLi_given_rest)

    return L

def gibbs_sample_A(network, L, params, burn_in=200):

    A = np.random.binomial(1, 0.5, len(network))

    # keep sampling an A vector till burn in is done
    for m in range(burn_in):
        for i in range(len(network)):
            pAi_given_rest = expit(params[0] + params[1]*L[i] +
                                   params[2]*np.dot(A, network[i, :]) -
                                   params[3]*np.dot(L, network[i, :]))
            A[i] = np.random.binomial(1, pAi_given_rest)

    return A

def gibbs_sample_Y(network, L, A, params, burn_in=200):

    Y = np.random.binomial(1, 0.5, len(network))

    # keep sampling an Y vector till burn in is done
    for m in range(burn_in):
        for i in range(len(network)):
            pYi_given_rest = expit(params[0] + params[1]*L[i] + params[2]*A[i] +
                                   params[3]*np.dot(A, network[i, :]) -
                                   params[4]*np.dot(L, network[i, :]) +
                                   params[5]*np.dot(Y, network[i, :]))
            Y[i] = np.random.binomial(1, pYi_given_rest)

    return Y

def npll_L(params, L, network):

    pL1 = expit(params[0] + params[1]*(L@network))
    pL = L*pL1 + (1-L)*(1-pL1)
    pL = np.where(pL == 0, 1e-10, pL) # replace 0 with a small const to ensure numerical stability
    return -np.sum(np.log(pL))

def npll_Y(params, L, A, Y, network):

    pY1 = expit((params[0] + params[1]*L + params[2]*A + params[3]*(A@network) + params[4]*(L@network) + params[5]*(Y@network)))
    pY = Y*pY1 + (1-Y)*(1-pY1)
    pY = np.where(pY == 0, 1e-10, pY) # replace 0 with a small const to ensure numerical stability
    return -np.sum(np.log(pY))

def estimate_causal_effects(network, A_value, params_L, params_Y, burn_in=100, K=100, N=3):
    '''
    K: number of rows in matrix_Ys
    N: thin the Markov Chain for every N iteration
    '''
    # initialize random values
    Y = np.random.binomial(1, 0.5, len(network))
    L = np.random.binomial(1, 0.5, len(network))

    A = [A_value] * len(L)  # a list of A_value repeated len(L) times
    matrix_Ys = []

    for m in range(burn_in + K*N):
        for i in range(len(network)):
            pLi_given_rest = expit(params_L[0] + params_L[1]*np.dot(L, network[i, :]))
            L[i] = np.random.binomial(1, pLi_given_rest)

            pYi_given_rest = expit(params_Y[0] + params_Y[1]*L[i] + params_Y[2]*A[i] +
                                   params_Y[3]*np.dot(A, network[i, :]) -
                                   params_Y[4]*np.dot(L, network[i, :]) +
                                   params_Y[5]*np.dot(Y, network[i, :]))
            Y[i] = np.random.binomial(1, pYi_given_rest)

        if m > burn_in and m % N == 0:
            matrix_Ys.append(Y.copy())

    return np.mean(matrix_Ys)

def bootstrap_causal_effect_estimates(n_units_list, n_bootstraps, true_L, true_A, true_Y, burn_in):

    estimates = {}

    for n_units in n_units_list:

        estimates_n_units = []
        for _ in range(n_bootstraps):
            network = random_network_adjacency_matrix(n_units, 1, 6)

            # create a single network realization using true parameters
            L = gibbs_sample_L(network, params=true_L, burn_in=burn_in)
            A = gibbs_sample_A(network, L, params=true_A, burn_in=burn_in)
            Y = gibbs_sample_Y(network, L, A, params=true_Y, burn_in=burn_in)

            # estimate parameters of the DGP
            params_L = minimize(npll_L, x0=np.random.uniform(-1, 1, 2), args=(L, network)).x
            params_Y = minimize(npll_Y, x0=np.random.uniform(-1, 1, 6), args=(L, A, Y, network)).x
            # print("params L errors", np.abs(params_L-true_L))
            # print("params Y errors", np.abs(params_Y-true_Y))

            # evaluate estimated network causal effects
            Y_A1 = estimate_causal_effects(network, 1, params_L, params_Y, burn_in=burn_in) # beta alpha with A=1
            Y_A0 = estimate_causal_effects(network, 0, params_L, params_Y, burn_in=burn_in) # beta alpha with A=0
            estimates_n_units.append(Y_A1 - Y_A0)

        estimates[n_units] = estimates_n_units

    return estimates

def bootstrap_iteration(args):
    # decompose args
    n_units, true_L, true_A, true_Y, burn_in = args

    # create a single network realization using true parameters
    network = random_network_adjacency_matrix(n_units, 1, 6)
    L = gibbs_sample_L(network, params=true_L, burn_in=burn_in)
    A = gibbs_sample_A(network, L, params=true_A, burn_in=burn_in)
    Y = gibbs_sample_Y(network, L, A, params=true_Y, burn_in=burn_in)

     # estimate parameters of the DGP
    params_L = minimize(npll_L, x0=np.random.uniform(-1, 1, 2), args=(L, network)).x
    params_Y = minimize(npll_Y, x0=np.random.uniform(-1, 1, 6), args=(L, A, Y, network)).x

    # evaluate estimated network causal effects
    Y_A1 = estimate_causal_effects(network, 1, params_L, params_Y, burn_in=burn_in)
    Y_A0 = estimate_causal_effects(network, 0, params_L, params_Y, burn_in=burn_in)
    
    return Y_A1 - Y_A0

def bootstrap_causal_effect_estimates_parallel(n_units_list, n_bootstraps, true_L, true_A, true_Y, burn_in):

    estimates = {}
    with ProcessPoolExecutor() as executor:
        for n_units in n_units_list:
            args = [(n_units, true_L, true_A, true_Y, burn_in) for _ in range(n_bootstraps)]
            results = executor.map(bootstrap_iteration, args)
            estimates[f'n units {n_units}'] = list(results)

    return estimates

def main():
    # set up
    n_units_true_causal_effect = 5000
    n_bootstraps = 100
    n_units_list = [3000, 4000, 5000]
    burn_in = 200

    # evaluate true network causal effects
    network = random_network_adjacency_matrix(n_units_true_causal_effect, 1, 6)

    true_L = np.array([-0.3, 0.4])
    true_A = np.array([0.3, -0.4, -0.7, -0.2])
    true_Y = np.array([-0.2, 1, -1.5, 0.4, -0.3, 0.4])

    Y_A1 = estimate_causal_effects(network, A_value=1, params_L=true_L, params_Y=true_Y, burn_in=200, K=100, N=3)
    Y_A0 = estimate_causal_effects(network, A_value=0, params_L=true_L, params_Y=true_Y, burn_in=200, K=100, N=3)
    true_causal_effect = Y_A1 - Y_A0
    print("True Causal Effects:", true_causal_effect)

    # Using the parallelized function for auto-g estimation
    est_causal_effects = bootstrap_causal_effect_estimates_parallel(
        n_units_list=n_units_list, 
        n_bootstraps=n_bootstraps, 
        true_L=true_L, 
        true_A=true_A, 
        true_Y=true_Y, 
        burn_in=burn_in
    )

    df = pd.DataFrame.from_dict(est_causal_effects, orient='index').transpose()
    df['True Effect'] = true_causal_effect
    df.to_csv("./autog_results.csv", index=False)
    print(f"Results saved.")

if __name__ == "__main__":
    main()
