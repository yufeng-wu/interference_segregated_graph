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

def biedge_sample_L(network, params):
    
    U = np.random.normal(loc=params[0], scale=params[1], size=network.shape)
    U = np.triu(U) + np.triu(U, 1).T # make U symmetric by copying the upper triangular to the lower triangular part
    U = np.where(network == 1, U, network) # apply the network mask

    pL = expit(params[2] + params[3]*U.sum(axis=0)) # pL is a vector 
    L = np.random.binomial(1, pL)

    return L

def biedge_sample_A(network, L, params):

    U = np.random.normal(loc=params[0], scale=params[1], size=network.shape)
    U = np.triu(U) + np.triu(U, 1).T  # make U symmetric
    U = np.where(network == 1, U, network)  # apply network mask

    pA = expit(params[2] + params[3]*L + params[4]*(L@network) + params[5]*U.sum(axis=0))
    A = np.random.binomial(1, pA)

    return A

def biedge_sample_Y(network, L, A, params):

    U = np.random.normal(loc=params[0], scale=params[1], size=network.shape)
    U = np.triu(U) + np.triu(U, 1).T  # make U symmetric
    U = np.where(network == 1, U, network)  # apply network mask
    print("Avg of U.sum:", np.mean(U.sum(axis=0)))
    pY = expit(params[2] + params[3]*L + params[4]*A + params[5]*(L@network) + 
               params[6]*(A@network) + params[7]*U.sum(axis=0))
    Y = np.random.binomial(1, pY)

    return Y

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
                                   params[2]*np.dot(A, network[i, :]) +
                                   params[3]*np.dot(L, network[i, :]))
            A[i] = np.random.binomial(1, pAi_given_rest)

    return A

def gibbs_sample_Y(network, L, A, params, burn_in=200):

    Y = np.random.binomial(1, 0.5, len(network))

    # keep sampling an Y vector till burn in is done
    for m in range(burn_in):
        for i in range(len(network)):
            pYi_given_rest = expit(params[0] + params[1]*L[i] + params[2]*A[i] +
                                   params[3]*np.dot(L, network[i, :]) +
                                   params[4]*np.dot(A, network[i, :]) +
                                   params[5]*np.dot(Y, network[i, :]))
            Y[i] = np.random.binomial(1, pYi_given_rest)

    return Y

def npll_L(params, L, network):

    pL1 = expit(params[0] + params[1]*(L@network))
    pL = L*pL1 + (1-L)*(1-pL1)
    pL = np.where(pL == 0, 1e-10, pL) # replace 0 with a small const to ensure numerical stability
    return -np.sum(np.log(pL))

def npll_Y(params, L, A, Y, network):

    pY1 = expit((params[0] + params[1]*L + params[2]*A + params[3]*(L@network) + params[4]*(A@network) + params[5]*(Y@network)))
    pY = Y*pY1 + (1-Y)*(1-pY1)
    pY = np.where(pY == 0, 1e-10, pY)
    return -np.sum(np.log(pY))

def estimate_causal_effects_U_U(network, A_value, params_L, params_Y, 
                                burn_in=200, K=100, N=3):
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
                                   params_Y[3]*np.dot(L, network[i, :]) +
                                   params_Y[4]*np.dot(A, network[i, :]) +
                                   params_Y[5]*np.dot(Y, network[i, :]))
            Y[i] = np.random.binomial(1, pYi_given_rest)

        if m > burn_in and m % N == 0:
            matrix_Ys.append(Y.copy())

    return np.mean(matrix_Ys)

def estimate_causal_effects_B_B(network, A_value, params_L, params_Y, K=100):
    matrix_Ys = []

    for _ in range(K):
        L = biedge_sample_L(network, params_L)
        A = np.array([A_value] * len(L))
        Y = biedge_sample_Y(network, L, A, params_Y)

        matrix_Ys.append(Y.copy())
    
    return np.mean(matrix_Ys)

def estimate_causal_effects_U_B(network, A_value, params_L, params_Y, 
                                burn_in=200, K=100):
    matrix_Ys = []

    for _ in range(K):
        L = gibbs_sample_L(network, params_L, burn_in)
        A = np.array([A_value] * len(L))
        Y = biedge_sample_Y(network, L, A, params_Y)

        matrix_Ys.append(Y.copy())
    
    return np.mean(matrix_Ys)

def _autog(args):
    ''' 
    Generate a network realization following L_edge_type, A_edge_type, and 
    Y_edge_type, then estimate causal effects using auto-g.
    
    This function is used to demonstrate that auto-g produces consistent 
    estimates of network causal effects when the DGP follows the "UUU" 
    specification, but it produces biased estimates when there is 
    latent homophily at the L or the Y layer. 
    '''
    n_units, L_edge_type, A_edge_type, Y_edge_type, true_L, true_A, true_Y, burn_in = args

    # create a single network realization using the true parameters
    network = random_network_adjacency_matrix(n_units, 1, 6)
    if L_edge_type == "U":
        L = gibbs_sample_L(network, params=true_L, burn_in=burn_in)
    elif L_edge_type == "B":
        L = biedge_sample_L(network, params=true_L)

    if A_edge_type == "U":
        A = gibbs_sample_A(network, L, params=true_A, burn_in=burn_in)
    elif A_edge_type == "B":
        A = biedge_sample_A(network, L, params=true_A)

    if Y_edge_type == "U":
        Y = gibbs_sample_Y(network, L, A, params=true_Y, burn_in=burn_in)
    elif Y_edge_type == "B":
        Y = biedge_sample_Y(network, L, A, params=true_Y)

    # use L, A, Y to estimate parameters using auto-g regardless of whether the 
    # true edge types are UUU or not.
    params_L = minimize(npll_L, x0=np.random.uniform(-1, 1, 2), 
                        args=(L, network)).x
    params_Y = minimize(npll_Y, x0=np.random.uniform(-1, 1, 6), 
                        args=(L, A, Y, network)).x

    # compute causal effects using estimated parameters
    Y_A1 = estimate_causal_effects_U_U(network, 1, params_L, params_Y, 
                                       burn_in=burn_in)
    Y_A0 = estimate_causal_effects_U_U(network, 0, params_L, params_Y, 
                                       burn_in=burn_in)
    
    return Y_A1 - Y_A0

def bootstrap_autog(n_units_list, L_edge_type, A_edge_type, Y_edge_type, 
                    n_bootstraps, true_L, true_A, true_Y, burn_in):
    '''
    Bootstrap a confidence interval of causal effects computed using auto-g, 
    with data generated from a graphical model specified with L_edge_type,
    A_edge_type, and Y_edge_type.
    '''
    estimates = {}
    with ProcessPoolExecutor() as executor:
        for n_units in n_units_list:
            args = [(n_units, L_edge_type, A_edge_type, Y_edge_type, true_L, 
                     true_A, true_Y, burn_in) for _ in range(n_bootstraps)]
            results = executor.map(_autog, args)
            estimates[f'n units {n_units}'] = list(results)

    return estimates



def BBB_experiment():
    # set up
    n_units_true_causal_effect = 9000
    n_bootstraps = 100
    n_units_list = [1000, 3000, 5000, 7000, 9000]
    burn_in = 200
    
    # evaluate true network causal effects 
    network = random_network_adjacency_matrix(n_units_true_causal_effect, 1, 6)

    true_L = np.array([0, 1, -0.3, 0.4])
    true_A = np.array([0, 1, 0.3, -0.4, -0.7, 0.2])
    true_Y = np.array([0, 1, 0.5, 0.1, 1, -0.3, 0.6, 0.4])

    Y_A1 = estimate_causal_effects_B_B(network, 1, true_L, true_Y, K=50)
    Y_A0 = estimate_causal_effects_B_B(network, 0, true_L, true_Y, K=50)
    true_causal_effect = Y_A1 - Y_A0

    # Using the parallelized function for auto-g estimation (BBB)
    est_causal_effects = bootstrap_autog(
        L_edge_type="B",
        Y_edge_type="B",
        n_units_list=n_units_list, 
        n_bootstraps=n_bootstraps, 
        true_L=true_L, 
        true_A=true_A, 
        true_Y=true_Y,
        burn_in=burn_in
    )

    df = pd.DataFrame.from_dict(est_causal_effects, orient='index').transpose()
    df['True Effect'] = true_causal_effect
    df.to_csv("./autog_BBB_results.csv", index=False)
    print(f"Results saved.")

    # create a single network realization using true parameters
    # L = biedge_sample_L(network, true_L)
    # A = biedge_sample_A(network, L, true_A)
    # Y = biedge_sample_Y(network, L, A, true_Y)
    # print("means", np.mean(L), np.mean(A), np.mean(Y))

    # # estimate parameters of the DGP
    # est_params_L = minimize(npll_L, x0=np.random.uniform(-1, 1, 2), args=(L, network)).x
    # est_params_Y = minimize(npll_Y, x0=np.random.uniform(-1, 1, 6), args=(L, A, Y, network)).x

    # est_Y_A1 = estimate_causal_effects(network, 1, est_params_L, est_params_Y, 200, 100, 3)
    # est_Y_A0 = estimate_causal_effects(network, 0, est_params_L, est_params_Y, 200, 100, 3)
    # print("Estimated Causal Effects:", est_Y_A1 - est_Y_A0)




# TODO: check consistency of autog
# TODO: implement estimation strategy (predict Yi using A_Ni,i L_Ni,i using 1 hop data) for UBB and check consistency
# TODO: show that UBB and BBB case, when estimated using autog, is inconsistent

# Question: should the true causal effect of the UBB case andt the BBB case be the same?
# Question: for the UBB case, why does changing the coefficient of U change true causal effect?